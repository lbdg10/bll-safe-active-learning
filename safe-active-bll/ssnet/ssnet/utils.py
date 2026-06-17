from pathlib import Path

import scipy.io as sio


class MatFileWriter:
    def __init__(self, path: str | Path) -> None:
        """
        Auxiliary class that orderly manages the saving of variables to a MATLAB file.

        Parameters
        ----------
        path : str | Path
            The path of the mat file.
        """
        self.path = Path(path).with_suffix('.mat')
        self._buffer = {}

    def push(self, **kwds):
        """
        Queue data to save.

        Raises
        ------
        ValueError
            When a key is already in the buffer
        """
        for key, item in kwds.items():
            if key in self._buffer:
                raise ValueError(f'Item {key} already exists in the buffer')
            self._buffer[key] = item
    
    def close(self):
        """
        Save the mat file.
        """
        sio.savemat(self.path, self._buffer)
