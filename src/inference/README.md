# General

Here you can run inference for two pipelines:
- Speech to Face: Given and audio of a human speech, generate the face embedding of that person, and then generate face image out of that embedding
- Face to Face: Given an image of a face, generate face embedding, and then generate face image out of that embedding

## Speech to Face

In this pipeline two models are used:
- VoiceEncoder: It is used to generate 4096-D face feature vector out of speech spectrogram
- FaceDecoder: It is used to decode generated embedding by VoiceEncoder into an image of the speaker

You need to:
- pass `--audio-file-path` parameter, which specifies a path to the audio file. It must be a simple audio file. The script does all of the preprocessing to convert the audio to the spectrogram
- pass `--voice-encoder-path` parameter, which specifies a path to the trained weights of the VoiceEncoder model
- pass `--face-decoder-path` parameter, which specifies a path to the trained weights of the FaceDecoder model

Use `inference_speech_to_face_ast.py` script when you want to use AST as the VoiceEncoder or use `inference_speech_to_face_ve.py` script when you want to use the VE_conv architecture from the SpeechToFace paper. Scripts `inference_speech_to_face_ast_dataset.py` and `inference_speech_to_face_ve_dataset.py` are used to generate image from every audio file in a given directory specified by `--test-set-path` parameter.

## Face to Face

In this pipeline two models are used:
- FaceEncoder: It is used to generate 4096-D face feature vector out an image of the person face
- FaceDecoder: It is used to decode generated embedding by FaceEncoder into an image of the face

You need to:
- pass `--image-file-path` parameter, which specifies a path to the image file. Image can be of arbitrary size, because script does all of the preprocessing to convert the image to a Tensor of size 224x224 with 3 channels (RGB)
- pass `--face-encoder-model` parameter, which an arhitecture of the FaceEncoder (see `src/models` for more details)
- pass `--face-encoder-weights-path` parameter, which specifies a path to the trained weights of the chosen FaceEncoder model
- pass `--face-decoder-path` parameter, which specifies a path to the trained weights of the FaceDecoder model
