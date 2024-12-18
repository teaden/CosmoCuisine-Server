'''
FastAPI server code for CosmoCuisine iOS app

This server code is singular in purpose - to take .wav files from the client and classify them as languages using the
ECAPA-TDNN model pretrained on the VoxLingua107 Data Set (https://huggingface.co/TalTechNLP/voxlingua107-epaca-tdnn)
'''

# FastAPI imports
from fastapi import FastAPI, HTTPException, UploadFile, File

# Machine Learning Imports
from speechbrain.inference import EncoderClassifier

# Define some elements in the API
async def custom_lifespan(app: FastAPI):

    # Set up the ECAPA-TDNN classifier pre-trained on VoxLingua107
    app.language_id = EncoderClassifier.from_hparams(
        source="./Voxlingua107-ECAPA-TDNN",
        savedir="./Voxlingua107-ECAPA-TDNN")

    yield


# Create the FastAPI app
app = FastAPI(
    title="CosmoCuisine",
    summary="Find out nutrition facts about packaged goods using audio, vision, and ML!",
    lifespan=custom_lifespan,
)


#========================================
#   Data store objects from pydantic
#----------------------------------------

'''See pydantic_models.py
'''




#===========================================
#   Machine Learning methods (Scikit-learn)
#-------------------------------------------
# These allow us to interact with the REST server with ML from Scikit-learn.

@app.post(
    "/predict",
    response_description="Predict Label from Datapoint",
)
async def predict_datapoint(
        wav_file: UploadFile = File(...)
):
    """
    Post a feature data and get the prediction back
    """
    try:
        try:
            # Read .wav file uploaded as part of multipart/form-data request
            audio_content = await wav_file.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading wav file: {e}")

        try:
            # Create a temporary file path
            temp_filepath = "./recording.wav"

            # Write the uploaded bytes to the temporary file
            with open(temp_filepath, "wb") as f:
                f.write(audio_content)

            # Classify language using path to wave file per load_audio expectations
            signal = app.language_id.load_audio(temp_filepath)
            prediction = app.language_id.classify_batch(signal)
            predicted_label = prediction[3][0]
            return {"prediction": predicted_label}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error predicting with ML model: {e}")
    finally:
        await wav_file.close()
