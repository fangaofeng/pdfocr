import cv2
import base64
import numpy as np
import queue


from aip import AipOcr


def encode(img):
    cv2_encode = cv2.imencode('.jpg', img)[1].tostring()
    base64_image = base64.b64encode(cv2_encode)
    return base64_image


def decode(base64_image):
    # decode_img = base64.b64decode(base64_image)
    decode_img = np.fromstring(base64_img.decode('base64'), np.uint8)
    return decode_img


class OcrApiPool(object):

    def __init__(self, debug=False):
        # 创建一个队列，队列里最多只能有10个数据
        baiduapp = {'talbe1': {'APP_ID': '11507399', 'API_KEY': 'DH6shd00MjvNolrX3WskgNmC', 'SECRET_KEY': 'uO0FQ0P4c1zFQh0QvWBgeuFMfaKsRpq4'},
                    'talbe2': {'APP_ID': '14266691', 'API_KEY': 'IQeiCUn7In39sfiGu7P65aKq', 'SECRET_KEY': 'SHNF4thUhIU4DOBDZkNLn113nhgwodNr'}}
        self.queue = queue.Queue(len(baiduapp))
        # 在队列里填充线程类
        # 【线程类、线程类、线程类、线程类、线程类、线程类、线程类】
        for key, value in baiduapp.items():
            ocrapi = BaiduAipOcr(
                value['APP_ID'], value['API_KEY'], value['SECRET_KEY'], debug)
            self.queue.put(ocrapi)

    def get_ocrapi(self):
        # 去队列里去数据，
        # queue特性，如果有，对列里那一个出来
        #            如果没有，阻塞，
        return self.queue.get()

    def add_ocrapi(self, ocrapi):
        # 往队列里再添加一个线程类
        self.queue.put(ocrapi)


class BaiduAipOcr(AipOcr):

    def __init__(self, APP_ID='14266691', API_KEY='IQeiCUn7In39sfiGu7P65aKq', SECRET_KEY='SHNF4thUhIU4DOBDZkNLn113nhgwodNr', debug=False):
        self.APP_ID = APP_ID
        self.API_KEY = API_KEY
        self.SECRET_KEY = SECRET_KEY
        self.debug = debug
        super(BaiduAipOcr, self).__init__(APP_ID, API_KEY, SECRET_KEY)

    def basicGeneralforCv2(self, image, options=None):
        """
            通用文字识别
        """
        options = options or {}

        data = {}
        data['image'] = encode(image).decode()

        data.update(options)
        # return {"words_result": [{"words": "tett"}],"words_result_num": 1}
        return self._request(self._AipOcr__generalBasicUrl, data)

    def basicAccurateforCv2(self, image, options=None):
        """
            通用文字识别
        """
        options = options or {}

        data = {}
        data['image'] = encode(image).decode()

        data.update(options)
        # return {"words_result": [{"words": "tett"}],"words_result_num": 1}
        return self._request(self._AipOcr__accurateBasicUrl, data)

    def image_to_stringforcv2(self, image, options=None, typelist=False):
        text = ''
        result = self.basicGeneralforCv2(image, options)
        logid = result.get('log_id', None)
        if self.debug:
            print(result, self.APP_ID)

        if logid is None:
            return text
        words_result = result['words_result']
        line_num = result['words_result_num']
        if typelist:
            return [words_result[i]['words'] for i in range(line_num)]
        for i in range(line_num):
            text = ' '.join([text, words_result[i]['words']])
        return text

    def image_to_stringforfile(self, fileimg, options=None):
        text = ''
        result = self.basicGeneral(fileimg, options)
        logid = result.get('log_id', None)
        if self.debug:
            print(result, self.APP_ID)
        if logid is None:
            return text

        words_result = result['words_result']
        line_num = result['words_result_num']
        for i in range(line_num):
            text = ' '.join([text, words_result[i]['words']])
        return text

    def tableRecognitionAsyncforCv2(self, image, options=None):
        """
            通用文字识别
        """
        options = options or {}

        data = {}
        data['image'] = encode(image).decode()

        data.update(options)
        # return {"words_result": [{"words": "tett"}],"words_result_num": 1}
        return self._request(self._AipOcr__tableRecognizeUrl, data)


if __name__ == '__main__':
    import cv2
    APP_ID = '11507399'
    API_KEY = 'DH6shd00MjvNolrX3WskgNmC'
    SECRET_KEY = 'uO0FQ0P4c1zFQh0QvWBgeuFMfaKsRpq4'

    client = BaiduAipOcr()
    options = {}
    options["language_type"] = "CHN_ENG"
    image = cv2.imread('testimage/441130.jpg')

    """ 带参数调用通用文字识别（含位置信息版）, 图片参数为本地图片 """
    resutl = client.basicGeneralforCv2(image, options)
    print(resutl)
    """
    ori_img = cv2.imread('/home/tunnel/past/crop/BJN4988.jpg')
    base64_img = encode(ori_img)
    print(base64_img)
    decoded_img = decode(base64_img)
    rgb_img = cv2.imdecode(decoded_img, cv2.IMREAD_ANYCOLOR)
    cv2.imwrite("/home/tunnel/test.png", rgb_img)
    """
