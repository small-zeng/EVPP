import math

class Vector3d():
    """
    2维向量, 支持加减, 支持常量乘法(右乘)
    """

    def __init__(self, x):
        self.deltaX = x[0]
        self.deltaY = x[1]
        self.deltaZ = x[2]
        self.length = -1
        self.direction = [0, 0, 0]
        self.vector3d_share()

    def vector3d_share(self):
        if type(self.deltaX) == type(list()) and type(self.deltaY) == type(list()) and  type(self.deltaZ) == type(list()):
            print('输入超过1个向量')
            # deltaX, deltaY, deltaZ = self.deltaX, self.deltaY, self.deltaZ
            # self.deltaX = deltaY[0] - deltaX[0]
            # self.deltaY = deltaY[1] - deltaX[1]
            
            # self.length = math.sqrt(self.deltaX ** 2 + self.deltaY ** 2 + self.deltaZ ** 2) * 1.0
            # if self.length > 0:
            #     self.direction = [self.deltaX / self.length, self.deltaY / self.length]
            # else:
            #     self.direction = None
        else:
            self.length = math.sqrt(self.deltaX ** 2 + self.deltaY ** 2 + self.deltaZ ** 2) * 1.0
            if self.length > 0:
                self.direction = [self.deltaX / self.length, self.deltaY / self.length, self.deltaZ / self.length]
            else:
                self.direction = None

    def __add__(self, other):
        """
        + 重载
        :param other:
        :return:
        """
        vec = Vector3d([self.deltaX, self.deltaY, self.deltaZ])
        vec.deltaX += other.deltaX
        vec.deltaY += other.deltaY
        vec.deltaZ += other.deltaZ
        vec.vector3d_share()
        return vec

    def __sub__(self, other):
        vec = Vector3d([self.deltaX, self.deltaY, self.deltaZ])
        vec.deltaX -= other.deltaX
        vec.deltaY -= other.deltaY
        vec.deltaZ -= other.deltaZ
        vec.vector3d_share()
        return vec

    def __mul__(self, other):
        vec = Vector3d([self.deltaX, self.deltaY, self.deltaZ])
        vec.deltaX *= other
        vec.deltaY *= other
        vec.deltaZ *= other
        vec.vector3d_share()
        return vec

    def __truediv__(self, other):
        return self.__mul__(1.0 / other)

    def __repr__(self):
        return 'Vector deltaX:{}, deltaY:{}, deltaZ:{}, length:{}, direction:{}'.format(self.deltaX, self.deltaY,self.deltaZ, self.length,
                                                                             self.direction)