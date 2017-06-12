import ast

class TerpreTValue(object):
    def __init__(self, max_value, name=None, verbosity=0):
        self.__max_value = max_value
        self.__name = "<UNKNOWN>" if name is None else name
        self.__value = None
        self.__verbosity = 0

    def __log_(self, msg, lvl):
        if self.__verbosity > lvl:
            print msg

    def __log(self, msg): self.__log_(msg, 0)

    def __spew(self, msg): self.__log_(msg, 1)

    def __set(self, value):
        if value < 0 or value >= self.__max_value:
            raise Exception("Error: Value %i out of bounds [0..%i] for %s." % (value, self.__max_value - 1, type(self)))
        if self.__value is not None:
            raise Exception("Error: Trying to reset immutable value!")
        self.__log("Setting %s <- %i" % (self.__name, value))
        self.__value = value

    def set_to(self, value):
        if isinstance(value, int):
            self.__set(value)
        elif isinstance(value, TerpreTValue):
            self.__set(value.get())
        else:
            raise Exception("Error: Trying to set %s to unknown value %s (type %s)." % (self, str(value), type(value)))

    def get(self):
        if self.__value is None:
            raise Exception("Error: Trying to access unset value of %s." % (self))
        else:
            self.__spew("Getting %s (%i)" % (self.__name, self.__value))
            return self.__value

    def name(self):
        return self.__name


class Var(TerpreTValue):
    def __init__(self, max_value, name):
        TerpreTValue.__init__(self, max_value, name)

    def __str__(self):
        return ("Var %s" % self.name())


class Input(TerpreTValue):
    def __init__(self, max_value, name, value):
        TerpreTValue.__init__(self, max_value, name)
        super(Input, self).set_to(value)

    def set_to(self, value):
        raise Exception("Error: Trying to set Input value.")


class Output(TerpreTValue):
    def __init__(self, max_value, name):
        TerpreTValue.__init__(self, max_value, name)


def id_decorator(func):
    def func_wrapper(*args):
        print("Calling %s" % (func.func_name))
        return func(*args)
    return func_wrapper


def Runtime(inDom, outDom):
    return id_decorator


class TerpreTRuntime():
    def __init__(self, data):
        self.__data = data
        self.__outputs = {}

    def get_input(self, name, max_value):
        value = self.__data[name]
        return Input(max_value, name, value)

    def get_output(self, name, max_value):
        output = Output(max_value, name)
        self.__outputs[name] = output
        return output

    def check(self):
        result = True
        for (name, computed) in self.__outputs.iteritems():
            computed = computed.get()
            if name not in self.__data:
                print("Could not find expected output value for %s (computed %i)." % (name, computed))
                result = False
            else:
                expected = self.__data[name]
                if expected != computed:
                    print("Value for output %s is wrong (computed %i, expected %i)." % (name, computed, expected))
                    result = False
                # else:
                #     print("Value for output %s is correct (computed %i, expected %i)." % (name, computed, expected))

        return result
