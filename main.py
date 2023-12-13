import cv2
import os
import sys
import numpy as np

from cv2.typing import MatLike
from sklearn.cluster import KMeans

with open('file.json', 'w') as sys.stdout:

    class Img:
        original: MatLike
        input: str
        output: str
        kArr: [int]

        def __init__(self, input: str, output: str, kArr: [int]) -> None:
            self.input = input
            self.output = output
            self.kArr = kArr

        def load(self) -> None:
            self.original = cv2.imread(self.input)

        def save(self, fileName: str, data) -> None:
            cv2.imwrite(self.output + '/' + fileName, data)

        def getResolution(self):
            a, b, _ = self.original.shape
            return (lambda a, b: str(a) + 'x' + str(b))(a, b)


    class Handler:
        def __init__(self) -> None:
            self.count = 0

        def apply(self, img: Img) -> None:
            print('{', flush=True)

            path = 'out_k_'+str(img.kArr[self.count])+'.png'
            fullPath = img.output + '/' + path

            img.load()

            orgSize = os.path.getsize(img.input) / 1024

            print('\"resolution\": \"' + img.getResolution() + '\",')
            print('\"nOriginalColors\": ' + str(len(np.unique(img.original))) + ',')
            print('\"originalSize\": ' + str(orgSize) + ',')
            print('\"k\": ' + str(img.kArr[self.count]) + ',', flush=True)

            if (not os.path.exists(fullPath)):
                arr = img.original.reshape((-1, 3))

                result = KMeans(n_clusters=img.kArr[self.count], random_state=42, n_init='auto').fit(arr)

                centroides = result.cluster_centers_
                rotulos = result.labels_

                finalImg = centroides[rotulos].reshape(img.original.shape)
                img.save(path, finalImg)

            finalSize = os.path.getsize(fullPath) / 1024
            print('\"transformedSize\": ' + str(finalSize) + ",")
            print('\"sizeDiff\":' + str(orgSize - finalSize) + ",")
            print("\"isTransformedBigger\": " + ('true' if(orgSize < finalSize) else 'false') + ",")
            print('\"outputPath\": \"' + fullPath + '\"')
            print('}', flush=True)

            self.count += 1
            if (self.count < len(img.kArr)):
                print(',', flush=True)
                self.apply(img)


    kValues = [ 2, 8, 16, 32, 64, 128, 256 ]

    images = [
        Img('input/1.png', 'output/1', kValues),
        Img('input/2.png', 'output/2', kValues),
        Img('input/3.png', 'output/3', kValues),
        Img('input/4.png', 'output/4', kValues),
        Img('input/5.png', 'output/5', kValues),
        Img('input/6.png', 'output/6', kValues)
    ]

    print('[')
    for idx, img in enumerate(images):
        if not os.path.exists(img.output):
            os.mkdir(img.output)

        if (idx):
            print(',')

        print('{')
        print('\"image\": \"' + img.input, end="\",\n")
        print('\"results\": [', flush=True)
        handler = Handler()
        handler.apply(img)
        print("]")
        print('}', flush=True)
    print(']')