import os
import random
import torch
import ffmpeg
import hashlib
import librosa
import numpy as np
import soundfile as sf
import folder_paths
from huggingface_hub import hf_hub_download
from .uvr5.mdxnet import MDXNetDereverb
from .uvr5.vr import AudioPre, AudioPreDeEcho
from cuda_malloc import cuda_malloc_supported

input_path = folder_paths.get_input_directory()
output_path = folder_paths.get_output_directory()
base_path = os.path.dirname(input_path)
node_path = os.path.join(base_path,"custom_nodes/ComfyUI-UVR5")
weights_path = os.path.join(node_path, "uvr5")
# cuda_malloc_supported() misreports True on Apple Silicon, so detect the real
# backend with torch. Half precision is only safe on CUDA.
if torch.cuda.is_available():
    device = "cuda"
    is_half = True
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    device = "mps"
    is_half = False
else:
    device = "cpu"
    is_half = False


def audio_to_tempfile(audio):
    """Write a ComfyUI standard AUDIO dict to a temp wav file and return its path."""
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    # waveform: [batch, channels, samples] -> first item as [samples, channels]
    wav = waveform[0].cpu().numpy().T
    os.makedirs(input_path, exist_ok=True)
    tmp_path = os.path.join(input_path, "uvr5_input_%d.wav" % random.randint(0, 0xffffffff))
    sf.write(tmp_path, wav, sample_rate)
    return tmp_path


def file_to_audio(path):
    """Load an audio file into a ComfyUI standard AUDIO dict."""
    wav, sr = librosa.load(path, sr=None, mono=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 1:
        wav = wav[np.newaxis, :]  # [channels, samples]
    waveform = torch.from_numpy(wav).unsqueeze(0)  # [1, channels, samples]
    return {"waveform": waveform, "sample_rate": int(sr)}


class PreViewAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"audio": ("AUDIOPATH",),}
                }

    CATEGORY = "AIFSH_UVR5"
    DESCRIPTION = "hello world!"

    RETURN_TYPES = ()

    OUTPUT_NODE = True

    FUNCTION = "load_audio"

    def load_audio(self, audio):
        audio_name = os.path.basename(audio)
        tmp_path = os.path.dirname(audio)
        audio_root = os.path.basename(tmp_path)
        return {"ui": {"audio":[audio_name,audio_root]}}
        

    @classmethod
    def IS_CHANGED(s, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)
        m = hashlib.sha256()
        with open(audio_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()


class LoadAudioPath:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = input_path
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.split('.')[-1].lower() in ["wav", "mp3","flac","m4a"]]
        return {"required":
                    {"audio": (sorted(files),)},
                }

    CATEGORY = "AIFSH_UVR5"

    RETURN_TYPES = ("AUDIOPATH",)
    FUNCTION = "load_audio"

    def load_audio(self, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)
        print(audio_path)
        return (audio_path,)

    @classmethod
    def IS_CHANGED(s, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)
        m = hashlib.sha256()
        with open(audio_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()


class UVR5:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict): 
        Tell the main program input parameters of nodes.
    IS_CHANGED:
        optional method to control when the node is re executed.

    Attributes
    ----------
    RETURN_TYPES (`tuple`): 
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        model_list = ["HP5_only_main_vocal.pth","HP5-主旋律人声vocals+其他instrumentals.pth","HP5_only_main_vocal.pth",
                      "HP2_all_vocals.pth","HP2-人声vocals+非人声instrumentals.pth","HP3_all_vocals.pth",
                      "VR-DeEchoAggressive.pth","VR-DeEchoDeReverb.pth","VR-DeEchoNormal.pth","onnx_dereverb_By_FoxJoy"]
        return {
            "required": {
                "audio": ("AUDIO",),
                "model": (model_list,{
                    "default": "HP5-主旋律人声vocals+其他instrumentals.pth"
                }),
                "agg":("INT",{
                    "default": 10, 
                    "min": 0, #Minimum value
                    "max": 20, #Maximum value
                    "step": 1, #Slider's step
                    "display": "slider"
                }),
                "format0":(["wav", "flac", "mp3", "m4a"],{
                    "default": "wav"
                })
            },
        }

    RETURN_TYPES = ("AUDIO","AUDIO")
    RETURN_NAMES = ("vocal_AUDIO","bgm_AUDIO")

    FUNCTION = "split"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_UVR5"

    def split(self, audio, model,agg,format0):
        # Accept ComfyUI standard AUDIO: write it to a temp file for the path-based backend.
        audio = audio_to_tempfile(audio)

        if model == "onnx_dereverb_By_FoxJoy":
            if not os.path.isfile(os.path.join(weights_path,"uvr5_weights/onnx_dereverb_By_FoxJoy", "vocals.onnx")):
                hf_hub_download(
                    repo_id="lj1995/VoiceConversionWebUI",
                    filename="vocals.onnx",
                    subfolder= "uvr5_weights/onnx_dereverb_By_FoxJoy",
                    local_dir= weights_path
                )
        else:
            if not os.path.isfile(os.path.join(weights_path,"uvr5_weights", model)):
                hf_hub_download(
                    repo_id="lj1995/VoiceConversionWebUI",
                    filename=model,
                    subfolder= "uvr5_weights",
                    local_dir= weights_path
                )
        save_root_vocal = output_path
        save_root_ins = output_path
        try:
            vocal_path,bgm_path = self.uvr(model, audio, save_root_vocal,save_root_ins,agg, format0, device, is_half)
        except Exception as e:
            if device != "cpu":
                print(f"[UVR5] inference on '{device}' failed ({e}); retrying on CPU")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                vocal_path,bgm_path = self.uvr(model, audio, save_root_vocal,save_root_ins,agg, format0, "cpu", False)
            else:
                raise
        try:
            os.remove(audio)
        except:
            pass
        # Return ComfyUI standard AUDIO so it connects to SaveAudio/PreviewAudio directly.
        return (file_to_audio(vocal_path), file_to_audio(bgm_path),)

    def uvr(self, model_name, inp_root, save_root_vocal,save_root_ins, agg, format0, device=device, is_half=is_half):
        vocal_AUDIO,bgm_AUDIO = "", ""
        inp_root = inp_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        save_root_vocal = (
            save_root_vocal.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        save_root_ins = (
            save_root_ins.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        is_hp3 = "HP3" in model_name
        if model_name == "onnx_dereverb_By_FoxJoy":
            pre_fun = MDXNetDereverb(15)
        else:
            func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
            pre_fun = func(
                agg=int(agg),
                model_path=os.path.join(weights_path, "uvr5_weights",model_name),
                device=device,
                is_half=is_half,
            )
        inp_path = inp_root
        need_reformat = 1
        done = 0
        
        info = ffmpeg.probe(inp_path, cmd="ffprobe")
        if (
            info["streams"][0]["channels"] == 2
            and info["streams"][0]["sample_rate"] == "44100"
        ):
            need_reformat = 0
            vocal_AUDIO,bgm_AUDIO = pre_fun._path_audio_(
                inp_path, save_root_ins, save_root_vocal, format0,is_hp3
            )
            done = 1
        else:
            need_reformat = 1
            
        if need_reformat == 1:
            tmp_path = "%s/%s.reformatted.wav" % (
                input_path,
                os.path.basename(inp_path),
            )
            os.system(
                f'ffmpeg -i "{inp_path}" -vn -acodec pcm_s16le -ac 2 -ar 44100 "{tmp_path}" -y'
            )
            inp_path = tmp_path
        
        if done == 0:
            vocal_AUDIO,bgm_AUDIO = pre_fun._path_audio_(
                inp_path, save_root_ins, save_root_vocal, format0,is_hp3
            )
            print("%s->Success" % (os.path.basename(inp_path)))
        
        try:
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model
                del pre_fun
        except:
            pass
        print("clean_empty_cache")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return vocal_AUDIO,bgm_AUDIO
    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    
