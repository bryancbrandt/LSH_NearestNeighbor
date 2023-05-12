# Similarity Search in High Dimensions via Hashing Implementation
# Derived from the paper by Gionis, Indyk, Motwani, 1999
# Bryan Brandt
# CS738 2023

"""
@author: HP
https://github.com/sandeepPalagati/PriorityQueue/blob/master/PriorityQueue.py
"""



from MinHeap import MinHeap


class PriorityQueue:
    def __init__(self, withObject=False):
        self.__withObject = withObject

        self.object = MinHeap()

    # Adding the elements to the priority queue
    def add(self, key, value=None):
        if self.__withObject and value == None:
            raise Exception("Null Object is not accepted")
        elif (not self.__withObject) and (value != None):
            raise Exception("Cannot accept object inplace of primitive values")

        if not isinstance(key, (int, float)):
            raise Exception(str(type(key)) + " is not invalid as key, valid:(int,float)")

        self.object.add(key, value)

    # Getting the top element
    def peek(self):
        if len(self) > 0:
            return self.object.peek()
        raise Exception("RunTime Exception:PriorityQueue is Empty")

    # Getting the top element and removing it.
    def poll(self):
        if len(self) > 0:
            return self.object.poll()
        raise Exception("RunTime Exception:PriorityQueue is Empty")

    # getting the length of priority queue.
    def __len__(self):
        return len(self.object)

    # checking whteher a priority queue is empty or not.
    def isEmpty(self):
        return self.object.isEmpty()
