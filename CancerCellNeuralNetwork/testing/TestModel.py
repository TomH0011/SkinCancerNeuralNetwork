
class TestModel:
    def test_model():
        file_uploader = FileUpload()
        prediction, certainty = file_uploader.test_one_image(uploaded_file_path=r"C:\Users\tomjh\Desktop")
        file_uploader.final_score(prediction, certainty)