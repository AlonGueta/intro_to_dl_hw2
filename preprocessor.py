import multiprocessing
import random

import numpy as np
from scipy import ndimage
from collections import namedtuple

Args = namedtuple('Args', ['angle', 'coordinates', 'steps', 'tilt'])


class Worker(multiprocessing.Process):

    def __init__(self, jobs, result, training_data, batch_size):
        super().__init__()
        ''' Initialize Worker and it's members.
        
        Parameters
        ----------
        jobs: Queue
            A jobs Queue for the worker.
        result: Queue
            A results Queue for the worker to put it's results in.
        
        You should add parameters if you think you need to.
        '''
        raise NotImplementedError("To be implemented")

    @staticmethod
    def rotate(image, angle):
        """Rotate given image to the given angle

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        angle : int
            The angle to rotate the image

        Return
        ------
        An numpy array of same shape
        """

        def inner_rotate(__image):
            return ndimage.rotate(__image, angle, reshape=False)

        return Worker.reshape_and_run(image, inner_rotate)

    @staticmethod
    def shift(image, dx, dy):
        """Shift given image

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        dx : int
            The number of pixels to move in the x-axis
        dy : int
            The number of pixels to move in the y-axis

        Return
        ------
        An numpy array of same shape
        """

        def inner_shift(__image):
            res = np.zeros_like(__image, dtype=float)
            get_neg_or_none = lambda x: x if x < 0 else None
            get_pos_or_none = lambda x: x if x > 0 else None

            res[get_pos_or_none(-dy): get_neg_or_none(-dy), get_pos_or_none(-dx): get_neg_or_none(-dx)] = \
                __image[get_pos_or_none(dy): get_neg_or_none(dy), get_pos_or_none(dx): get_neg_or_none(dx)]
            return res

        return Worker.reshape_and_run(image, inner_shift)

    @staticmethod
    def step_func(image, steps):
        """Transform the image pixels acording to the step function

                Parameters
                ----------
                image : numpy array
                    An array of shape 784 of pixels
                steps : int
                    The number of steps between 0 and 1

                Return
                ------
                An numpy array of same shape
                """

        def inner_step_function(__image):
            def step(x):
                return (1 / (steps - 1)) * np.floor(steps * x)

            step = np.vectorize(step, otypes=[np.float])
            return step(__image)

        return Worker.reshape_and_run(image, inner_step_function)

    @staticmethod
    def skew(image, tilt):
        """Skew the image

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        tilt : float
            The skew paramater

        Return
        ------
        An numpy array of same shape
        """

        def inner_skew(__image):
            res = np.zeros_like(__image)
            row, col = __image.shape[:2]
            for x in range(row):
                for y in range(col):
                    match_row = x + round(y * tilt)
                    res[x, y] = __image[match_row, y] if 0 <= match_row < row else 0.
            return res

        return Worker.reshape_and_run(image, inner_skew)

    def process_image(self, image):
        """Apply the image process functions

        Parameters
        ----------
        image: numpy array
            An array of size 784 of pixels

        Return
        ------
        An numpy array of same shape
        """
        (do_rotate, do_shift, do_step, do_skew) = (random.choice([True, False]) for _ in range(4))
        worker_args = Args(random.randint(-90, 90),  # angle between -90 to 90
                           (random.randint(-3, 3) for _ in range(2)),  # dx\dy between -5 to 5
                           random.randint(2, 10),  # steps between 2 to 10
                           random.uniform(-1.0, 1.0))  # tilt between -1.0 to 1.0 include the edges.
        worker_functions = [lambda __image: Worker.rotate(__image, worker_args.angle) if do_rotate else __image,
                            lambda __image: Worker.shift(__image, *worker_args.coordinates) if do_shift else __image,
                            lambda __image: Worker.step_func(__image, worker_args.steps) if do_step else __image,
                            lambda __image: Worker.skew(__image, worker_args.tilt) if do_skew else __image]

        functions_to_apply = []
        for i in range(4):
            f1 = random.choice(worker_functions)
            worker_functions.remove(f1)
            functions_to_apply.append(f1)
        res = np.copy(image)
        for func in functions_to_apply:
            res = func(res)

        return res

    def run(self):
        """Process images from the jobs queue and add the result to the result queue.
        """
        raise NotImplementedError("To be implemented")

    @staticmethod
    def reshape_and_run(image, func):
        image_matrix = np.reshape(image, (28, 28))
        res = func(image_matrix)
        return np.reshape(res, 784)
