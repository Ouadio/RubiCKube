
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict

NumAlphaDict = Dict[int, str]
AlphaNumDict = Dict[str, int]


class RubikCube:

    ALPHA_NUM_MAP: AlphaNumDict = {
        'F': 0, 'U': 1, 'R': 2, 'D': 3, 'L': 4, 'B': 5}
    NUM_ALPHA_MAP: NumAlphaDict = {
        0: 'F', 1: 'U', 2: 'R', 3: 'D', 4: 'L', 5: 'B'}
    NUM_COL_MAP: NumAlphaDict = {0: 'green', 1: 'white',
                                 2: 'red', 3: 'yellow', 4: 'orange', 5: 'blue'}

    # FaceId:{Neighboor:[index, axis, orientationSign]}
    NEIBOORING_MAP = {0: {1: [-1, 0, 1], 2: [0, 1, 1], 3: [0, 0, -1], 4: [-1, 1, -1]},
                      1: {5: [0, 0, -1], 2: [0, 0, -1], 0: [0, 0, -1], 4: [0, 0, -1]},
                      2: {1: [-1, 1, -1], 5: [0, 1, -1], 3: [-1, 1, -1], 0: [-1, 1, -1]},
                      3: {0: [-1, 0, 1], 2: [-1, 0, 1], 5: [-1, 0, -1], 4: [-1, 0, 1]},
                      4: {1: [0, 1, 1], 0: [0, 1, 1], 3: [0, 1, 1], 5: [-1, 1, 1]},
                      5: {3: [-1, 0, 1], 2: [-1, 1, -1], 1: [0, 0, -1], 4: [0, 1, 1]}}

    def __init__(self, dim: int = 3) -> None:
        # Cube dimension (max = 4 for full support)
        self._N = dim
        # Cube Faces : a dictionary of 0->5 keys, with _N*_N numpy array values
        # Each face data layout follows the standard visual layout in the standard
        # 2D flattening of the Cube
        self.__faces = dict([(k, k*np.ones([self._N, self._N], dtype='int16'))
                            for k in range(6)])

        # Whether the cube is ordered
        self._solved = True

        # Count of unit moves applied
        self._moves_count = 0

        # _algorithm as string (should be a LIFO, last is right side)
        self._algorithm = ""

        # More like a static attribute (TODO : fix it/cache it/hard code it)
        possible_moves = np.array(["F", "U", "R", "L", "D", "B"])
        if dim == 4:
            upcase_vec = np.vectorize(lambda sr: sr.lower())
            possible_moves = np.concatenate(
                (possible_moves, upcase_vec(possible_moves)))
        reverser_vec = np.vectorize(lambda sr: sr + "'")
        doubler_vec = np.vectorize(lambda sr: sr + "2")
        possible_moves = np.concatenate(
            (possible_moves, reverser_vec(possible_moves)))
        self._possible_moves = np.concatenate((
            possible_moves, doubler_vec(possible_moves[:len(possible_moves)//2])))

    def get_face(self, face_id: int) -> np.array:
        assert (face_id in range(0, 6)), "face id should be in [0:6["
        return self.__faces.get(face_id).copy()

    def print_face(self, face_id: int) -> None:
        assert (face_id in range(0, 6)), "face id should be in [0:6["
        print(f"FaceId {face_id}:")
        tempFace = self.__faces.get(face_id)
        for i in range(self._N):
            print(f"{tempFace[i,:]}")

    def print_all_faces(self) -> None:
        for k in self.__faces:
            self.print_face(k)

    def display_face(self, face_id: int, ax) -> None:
        """_summary_

        Args:
            face_id (int): Face Id
            ax (_type_): pyplot axis
        """
        # ax = plt.gca()
        face = self.__faces.get(face_id).copy()

        for i in range(self._N):
            for j in range(self._N):
                cellCol = RubikCube.NUM_COL_MAP[face[i, j]]
                square = patches.Rectangle(
                    (j, self._N-1-i), 1, 1, facecolor=cellCol, edgecolor='black', linewidth=2)
                ax.add_patch(square)

        plt.xlim(0, self._N)
        plt.ylim(0, self._N)

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.set_xticks([])
        ax.set_yticks([])

    def display(self) -> None:
        """
            Display in standard flattened layout the current state of the cube.
        """

        fig, axs = plt.subplots(3, 4)  # 3 rows, 4 columns

        # Display layout following international standard
        self.display_face(0, axs[1, 1])
        self.display_face(1, axs[0, 1])
        self.display_face(2, axs[1, 2])
        self.display_face(3, axs[2, 1])
        self.display_face(4, axs[1, 0])
        self.display_face(5, axs[1, 3])

        for ax in axs.flat:
            ax.set_xlim(0, self._N)
            ax.set_ylim(0, self._N)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        plt.tight_layout()
        plt.subplots_adjust(wspace=.05, hspace=.05)
        plt.show()

    def reset(self) -> None:
        """
            Reset Cube's into original solved state and re-initialize algorithm and counters.
        """
        for f in self.__faces:
            self.__faces[f] = np.ones(
                [self._N, self._N], dtype='int16')*f

        # Reset state values
        self._solved = True
        self._moves_count = 0
        self._algorithm = ""
        print("Cube reset successfully!")

    # ROTATION UTILS
    def __get_adj_slice(self, face_id: int, adj_id: int, layer_idx: int = 0) -> np.array:
        """
            Returns face-layer adjacent adj_id
        Args:
            face_id (int): Face Id
            adj_id (int): Adjacent Face id
            layer_idx (int, optional): Layer id starting face = 0. Defaults to 0.

        Returns:
            np.array : Array of _N ordered/oriented values contained in the adjacent slice.
        """

        assert (face_id >= 0 and face_id < 6), "face id should be in [0:6["

        slices = RubikCube.NEIBOORING_MAP[face_id]
        assert (adj_id in slices), f"face_id is not adjacent to {adj_id}"

        index = slices[adj_id][0]  # axisVal
        indexSign = np.sign(index+0.01)
        axis = slices[adj_id][1]  # axisDim, [x,:] if 0, [:,x] if 1
        sign = slices[adj_id][2]  # Orientation sign (clock-wise)
        index_ = int(index + indexSign*layer_idx)

        if axis == 0:
            return self.__faces[adj_id][index_, ::sign]
        else:
            return self.__faces[adj_id][::sign, index_]

    def __set_adj_slice(self, face_id: int, adj_id: int, layer_idx: int, slice_val: np.array) -> None:

        assert (face_id >= 0 and face_id < 6), "face id should be in [0:6["

        slices = RubikCube.NEIBOORING_MAP[face_id]
        assert (adj_id in slices), f"face_id is not adjacent to {adj_id}"

        index = slices[adj_id][0]  # axisVal
        indexSign = np.sign(index+0.01)
        axis = slices[adj_id][1]  # axisDim, [x,:] if 0, [:,x] if 1
        sign = slices[adj_id][2]  # Orientation sign (clock-wise)
        index_ = int(index + indexSign*layer_idx)

        if axis == 0:
            self.__faces[adj_id][index_, ::sign] = slice_val
        else:  # axis == 1
            self.__faces[adj_id][::sign, index_] = slice_val

    def __parse_instruction(self, instruction: str) -> dict:
        """_summary_

        Args:
            instruction (str): A valid single Rubik's Cube instruction

        Returns:
            dict: dict of {"face_id", "angle", "layer"} corresponding operation
        """

        assert instruction in self._possible_moves, f"Unvalid instruction {instruction}"

        if instruction.count("'"):
            axis_str = re.split("'", instruction)
            angle_sign = -1
        else:
            axis_str = instruction
            angle_sign = 1

        assert len(axis_str) < 3, "Invalid instruction length"

        if axis_str.count("2"):
            angle = 180
        else:
            angle = 90 * angle_sign

        face_str = axis_str[0]

        # Getting the int face id from its alphabetic notation
        face_id = RubikCube.ALPHA_NUM_MAP.get(face_str.upper())

        # When Cube dim = 4, lower case face refers to the 2nd slice
        layer = 1 - int(face_str.isupper())

        return {"face_id": face_id, "angle": angle, "layer": layer}

    def __rotate_layer(self, face_id: int,  angle: int, layer: int = 0) -> None:
        """ Rotates a given Cube Face identified by its 'face_id'
        degrees time, while reflecting the resulting changes on the
        adjacent faces. 

        Args:
            face_id (int): Id of cube face to be rotated 
            angle (signed int): Rotation angle in degrees (> 0 : right, 
                                < 0 : left)
            layer (unsigned int): from face_id perspective, index of the plane 
                                to rotate
        """

        assert (face_id >= 0 and face_id < 6), "face id should be in [0:6["

        # Adjacent faces Ids (Lock-wise order)
        adjacents = list(RubikCube.NEIBOORING_MAP[face_id].keys())

        # Adjacent slices to be read/written
        slice0 = self.__get_adj_slice(
            face_id=face_id, adj_id=adjacents[0], layer_idx=layer).copy()
        slice1 = self.__get_adj_slice(
            face_id=face_id, adj_id=adjacents[1], layer_idx=layer).copy()
        slice2 = self.__get_adj_slice(
            face_id=face_id, adj_id=adjacents[2], layer_idx=layer).copy()
        slice3 = self.__get_adj_slice(
            face_id=face_id, adj_id=adjacents[3], layer_idx=layer).copy()

        # Unit anti-transpose :
        InvUnit_ = np.identity(self._N, dtype='int16')[:, ::-1]

        # Rotation Parameters
        right_ = np.sign(angle) == 1
        degree_ = np.abs(int(angle)) % 360

        if degree_ == 90:
            if right_:
                # Turn right once
                if layer == 0:
                    self.__faces[face_id] = np.dot(
                        self.__faces[face_id].transpose(), InvUnit_)
                if layer == (self._N-1):
                    self.__faces[adjacents[4]] = self.__faces[adjacents[4]] = np.dot(
                        self.__faces[adjacents[4]], InvUnit_).transpose()
                self.__set_adj_slice(
                    face_id=face_id, adj_id=adjacents[0], layer_idx=layer, slice_val=slice3)
                self.__set_adj_slice(
                    face_id=face_id, adj_id=adjacents[1], layer_idx=layer, slice_val=slice0)
                self.__set_adj_slice(
                    face_id=face_id, adj_id=adjacents[2], layer_idx=layer, slice_val=slice1)
                self.__set_adj_slice(
                    face_id=face_id, adj_id=adjacents[3], layer_idx=layer, slice_val=slice2)
            else:
                # Turn left once
                if layer == 0:
                    self.__faces[face_id] = np.dot(
                        self.__faces[face_id], InvUnit_).transpose()
                if layer == (self._N-1):
                    self.__faces[adjacents[4]] = np.dot(
                        self.__faces[adjacents[4]].transpose(), InvUnit_)
                self.__set_adj_slice(
                    face_id=face_id, adj_id=adjacents[0], layer_idx=layer, slice_val=slice1)
                self.__set_adj_slice(
                    face_id=face_id, adj_id=adjacents[1], layer_idx=layer, slice_val=slice2)
                self.__set_adj_slice(
                    face_id=face_id, adj_id=adjacents[2], layer_idx=layer, slice_val=slice3)
                self.__set_adj_slice(
                    face_id=face_id, adj_id=adjacents[3], layer_idx=layer, slice_val=slice0)
        elif degree_ == 180:
            # Turn right (or left) twice
            if layer == 0:
                self.__faces[face_id] = self.__faces[
                    face_id][::-1, ::-1]
            if layer == (self._N-1):
                self.__faces[adjacents[4]] = self.__faces[
                    adjacents[4]][::-1, ::-1]
            self.__set_adj_slice(
                face_id=face_id, adj_id=adjacents[0], layer_idx=layer, slice_val=slice2)
            self.__set_adj_slice(
                face_id=face_id, adj_id=adjacents[1], layer_idx=layer, slice_val=slice3)
            self.__set_adj_slice(
                face_id=face_id, adj_id=adjacents[2], layer_idx=layer, slice_val=slice0)
            self.__set_adj_slice(
                face_id=face_id, adj_id=adjacents[3], layer_idx=layer, slice_val=slice1)
        elif degree_ == 0:
            return
        else:
            print("Only 90, 180 or 360 rotation angles supported so far.")
            return

    def __rotate_serie(self, serie: list) -> None:
        assert isinstance(serie, list), "Passed argument must be a list"
        assert isinstance(
            serie[0], dict), "Passed argument list must be a dictionary"
        serie_keys = list(serie[0].keys())
        assert ("face_id" in serie_keys) and ("angle" in serie_keys) and (
            "layer" in serie_keys), "Invalid Dictionaries"
        for step in serie:
            self.__rotate_layer(step.get("face_id"), step.get(
                "angle"), step.get("layer"))

    def apply_algorithm(self, str_serie: str) -> None:
        """_summary_

        Args:
            str_serie (str): Serie of operations (algorithm) separated by ; .
        """
        assert isinstance(str_serie, str), "Passed argument must be a string"

        if str_serie.count(";"):
            str_list = re.split(pattern=";", string=str_serie)
        else:
            str_list = [str_serie]
        steps = len(str_list)
        assert steps > 0, "List must contain at least one operation"

        serie = [dict()] * steps

        for i in range(steps):
            serie[i] = self.__parse_instruction(str_list[i])

        self.__rotate_serie(serie=serie)

        if self._solved:
            self._algorithm = str_serie
        else:
            self._algorithm += f";{str_serie}"

        self._moves_count += steps
        self._solved = False

    def get_alg_count(self) -> int:
        return (self._moves_count)

    def get_alg_log(self) -> str:
        return (self._algorithm)

    def is_solved(self) -> bool:
        """_summary_

        Returns:
            bool: True if the Cube is solved (each Cubie is in its default 
            position, case where overall permutation == Identity)
        """
        solved = True
        for i in range(6):
            solved_face = np.all(self.__faces.get(i).flatten() == i)
            solved = solved and solved_face
            if not solved:
                break

        return solved

    def scramble(self, steps: int) -> None:
        """_summary_

        Args:
            steps (int): Number of random unit moves to apply for scrambling 
            the cube.
        """

        rand_instr_list = np.random.choice(
            a=self._possible_moves, size=steps, replace=True)

        rand_instr_str = ";".join(rand_instr_list)
        self.apply_algorithm(rand_instr_str)

    def reverse_back(self, steps: int = 0) -> None:
        """
            Reverse back the current perumation of the cube steps-times. If 
            steps == 0, then revert all algorithm-history by applying counter
            operations (equivalent to reset() but without re-initializing the 
            algorithm history).
        Args:
            steps (int): Number of steps to reverse back.

        """
        assert steps <= self._moves_count, f"Can't reverse {steps} times as only {self._moves_count} have been made."

        # Reverse all algorithm history (backwards _moves_count time)
        if not steps:
            if self._moves_count:
                self.reverse_back(self._moves_count)
            else:
                print("No algorithm history found.")
            return

        alg_list = self._algorithm.split(";")
        alg_list = alg_list[:-(steps+1):-1]

        def reverse_it(st: str) -> str:
            if st.count("'"):
                return st[0]
            elif st.count("2"):
                return st
            else:
                return st + "'"
        alg_instr = ";".join([reverse_it(x) for x in alg_list])
        self.apply_algorithm(alg_instr)


if __name__ == "__main__":
    print("This is your Rubik's Cube")
