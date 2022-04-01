from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img # 이미지픽셀을 불러올 모듈


def deadSkin(data_dir, imgName):

    grid = 128

    imgData = load_img(f'{data_dir}/{imgName}', target_size= (grid, grid))

    model = load_model(f"{data_dir}/deadSkin_output_v1.h5")
    testimg = img_to_array(imgData)
    testimg = testimg/255

    h = model.predict(testimg.reshape(1, grid, grid, 3))

    print(f"각질 : {h.argmax()}")
