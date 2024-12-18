'''
Code for downloading the ECAPA-TDNN model pretrained on the VoxLingua107 Data Set
(https://huggingface.co/TalTechNLP/voxlingua107-epaca-tdnn)
'''

from speechbrain.inference import EncoderClassifier

if __name__ == '__main__':
    # Run the processing
    language_id = EncoderClassifier.from_hparams(
        source="TalTechNLP/voxlingua107-epaca-tdnn",
        savedir="Voxlingua107-ECAPA-TDNN"
    )
