"""
Define custom exceptions.


"""


class sdg4vsException(Exception):
    pass


class sdg4vsNanError(sdg4vsException):
    pass


class sdg4vsWrongParametrization(Exception):
    pass


# class a:
#     def __init__(self):
#         print("a initiated")

#     def init(self):
#         print("a")
#         self.test()

#     def test(self):
#         print("in a")


# class b(a):
#     def __init__(self):
#         a.__init__(self)
#         print("b initiated")

#     def init(self):
#         a.init(self)
#         print('b')

#     def test(self):
#         print("in b")


# x = b()

# x.init()
