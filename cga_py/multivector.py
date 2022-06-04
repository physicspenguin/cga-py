import numpy as np
import numba as nb


@nb.njit(cache=True)
def _mul(firstcoeff, othercoeff):  # pragma: no cover
    """CGA multiplication of two cga_object

    Parameters
    ----------
    other: cga_object
        cga_object to multiply with first
        Returns: (cga_object) multiplication of otherand first

    Returns
    -------

    """
    out = np.array(
        [
            (
                firstcoeff[0] * othercoeff[0]
                + firstcoeff[1] * othercoeff[1]
                + firstcoeff[2] * othercoeff[2]
                + firstcoeff[3] * othercoeff[3]
                + 2 * firstcoeff[9] * othercoeff[8]
                + 2 * firstcoeff[12] * othercoeff[11]
                + 2 * firstcoeff[14] * othercoeff[13]
                + 2 * firstcoeff[18] * othercoeff[17]
                + 2 * firstcoeff[20] * othercoeff[19]
                + 2 * firstcoeff[23] * othercoeff[22]
            )
            - (
                2 * firstcoeff[5] * othercoeff[4]
                + firstcoeff[6] * othercoeff[6]
                + firstcoeff[7] * othercoeff[7]
                + firstcoeff[10] * othercoeff[10]
                + firstcoeff[16] * othercoeff[16]
                + 2 * firstcoeff[27] * othercoeff[26]
            ),
            (
                firstcoeff[0] * othercoeff[1]
                + firstcoeff[1] * othercoeff[0]
                + 2 * firstcoeff[5] * othercoeff[8]
                + firstcoeff[6] * othercoeff[2]
                + firstcoeff[7] * othercoeff[3]
                + 2 * firstcoeff[12] * othercoeff[17]
                + 2 * firstcoeff[14] * othercoeff[19]
                + 2 * firstcoeff[18] * othercoeff[11]
                + 2 * firstcoeff[20] * othercoeff[13]
                + 2 * firstcoeff[27] * othercoeff[22]
            )
            - (
                firstcoeff[2] * othercoeff[6]
                + firstcoeff[3] * othercoeff[7]
                + 2 * firstcoeff[9] * othercoeff[4]
                + firstcoeff[10] * othercoeff[16]
                + firstcoeff[16] * othercoeff[10]
                + 2 * firstcoeff[23] * othercoeff[26]
            ),
            (
                firstcoeff[0] * othercoeff[2]
                + firstcoeff[1] * othercoeff[6]
                + firstcoeff[2] * othercoeff[0]
                + 2 * firstcoeff[5] * othercoeff[11]
                + firstcoeff[7] * othercoeff[16]
                + firstcoeff[10] * othercoeff[3]
                + 2 * firstcoeff[14] * othercoeff[22]
                + firstcoeff[16] * othercoeff[7]
                + 2 * firstcoeff[20] * othercoeff[26]
                + 2 * firstcoeff[23] * othercoeff[13]
            )
            - (
                firstcoeff[3] * othercoeff[10]
                + firstcoeff[6] * othercoeff[1]
                + 2 * firstcoeff[9] * othercoeff[17]
                + 2 * firstcoeff[12] * othercoeff[4]
                + 2 * firstcoeff[27] * othercoeff[19]
                + 2 * firstcoeff[18] * othercoeff[8]
            ),
            (
                firstcoeff[0] * othercoeff[3]
                + firstcoeff[1] * othercoeff[7]
                + firstcoeff[2] * othercoeff[10]
                + firstcoeff[3] * othercoeff[0]
                + 2 * firstcoeff[5] * othercoeff[13]
                + 2 * firstcoeff[27] * othercoeff[17]
            )
            - (
                firstcoeff[6] * othercoeff[16]
                + firstcoeff[7] * othercoeff[1]
                + 2 * firstcoeff[9] * othercoeff[19]
                + firstcoeff[10] * othercoeff[2]
                + 2 * firstcoeff[12] * othercoeff[22]
                + 2 * firstcoeff[14] * othercoeff[4]
                + firstcoeff[16] * othercoeff[6]
                + 2 * firstcoeff[18] * othercoeff[26]
                + 2 * firstcoeff[20] * othercoeff[8]
                + 2 * firstcoeff[23] * othercoeff[11]
            ),
            (
                firstcoeff[0] * othercoeff[4]
                + firstcoeff[1] * othercoeff[8]
                + firstcoeff[2] * othercoeff[11]
                + firstcoeff[3] * othercoeff[13]
                + firstcoeff[4] * othercoeff[0]
                + firstcoeff[26] * othercoeff[16]
                + 2 * firstcoeff[28] * othercoeff[17]
                + 2 * firstcoeff[29] * othercoeff[19]
                + 2 * firstcoeff[30] * othercoeff[22]
                + 2 * firstcoeff[31] * othercoeff[26]
            )
            - (
                2 * firstcoeff[25] * othercoeff[13]
                + 2 * firstcoeff[15] * othercoeff[4]
                + firstcoeff[6] * othercoeff[17]
                + firstcoeff[7] * othercoeff[19]
                + firstcoeff[8] * othercoeff[1]
                + firstcoeff[10] * othercoeff[22]
                + firstcoeff[11] * othercoeff[2]
                + firstcoeff[13] * othercoeff[3]
                + firstcoeff[16] * othercoeff[26]
                + firstcoeff[17] * othercoeff[6]
                + firstcoeff[19] * othercoeff[7]
                + firstcoeff[22] * othercoeff[10]
                + 2 * firstcoeff[21] * othercoeff[8]
                + 2 * firstcoeff[24] * othercoeff[11]
            ),
            (
                firstcoeff[0] * othercoeff[5]
                + firstcoeff[1] * othercoeff[9]
                + firstcoeff[2] * othercoeff[12]
                + firstcoeff[3] * othercoeff[14]
                + firstcoeff[5] * othercoeff[0]
                + 2 * firstcoeff[9] * othercoeff[21]
                + 2 * firstcoeff[12] * othercoeff[24]
                + 2 * firstcoeff[14] * othercoeff[25]
                + 2 * firstcoeff[18] * othercoeff[28]
                + 2 * firstcoeff[20] * othercoeff[29]
                + 2 * firstcoeff[23] * othercoeff[30]
                + firstcoeff[27] * othercoeff[16]
            )
            - (
                2 * firstcoeff[5] * othercoeff[15]
                + firstcoeff[6] * othercoeff[18]
                + firstcoeff[7] * othercoeff[20]
                + firstcoeff[9] * othercoeff[1]
                + firstcoeff[10] * othercoeff[23]
                + firstcoeff[12] * othercoeff[2]
                + firstcoeff[14] * othercoeff[3]
                + firstcoeff[16] * othercoeff[27]
                + firstcoeff[18] * othercoeff[6]
                + firstcoeff[20] * othercoeff[7]
                + firstcoeff[23] * othercoeff[10]
                + 2 * firstcoeff[27] * othercoeff[31]
            ),
            (
                firstcoeff[0] * othercoeff[6]
                + firstcoeff[1] * othercoeff[2]
                + firstcoeff[6] * othercoeff[0]
                + 2 * firstcoeff[9] * othercoeff[11]
                + firstcoeff[10] * othercoeff[7]
                + firstcoeff[16] * othercoeff[3]
                + 2 * firstcoeff[20] * othercoeff[22]
                + 2 * firstcoeff[27] * othercoeff[13]
                + 2 * firstcoeff[14] * othercoeff[26]
                + firstcoeff[3] * othercoeff[16]
            )
            - (
                firstcoeff[2] * othercoeff[1]
                + 2 * firstcoeff[5] * othercoeff[17]
                + firstcoeff[7] * othercoeff[10]
                + 2 * firstcoeff[12] * othercoeff[8]
                + 2 * firstcoeff[18] * othercoeff[4]
                + 2 * firstcoeff[23] * othercoeff[19]
            ),
            (
                firstcoeff[0] * othercoeff[7]
                + firstcoeff[1] * othercoeff[3]
                + firstcoeff[6] * othercoeff[10]
                + firstcoeff[7] * othercoeff[0]
                + 2 * firstcoeff[9] * othercoeff[13]
                + 2 * firstcoeff[23] * othercoeff[17]
            )
            - (
                firstcoeff[2] * othercoeff[16]
                + firstcoeff[3] * othercoeff[1]
                + 2 * firstcoeff[5] * othercoeff[19]
                + firstcoeff[10] * othercoeff[6]
                + 2 * firstcoeff[12] * othercoeff[26]
                + 2 * firstcoeff[14] * othercoeff[8]
                + firstcoeff[16] * othercoeff[2]
                + 2 * firstcoeff[18] * othercoeff[22]
                + 2 * firstcoeff[20] * othercoeff[4]
                + 2 * firstcoeff[27] * othercoeff[11]
            ),
            (
                firstcoeff[0] * othercoeff[8]
                + firstcoeff[1] * othercoeff[4]
                + firstcoeff[6] * othercoeff[11]
                + firstcoeff[7] * othercoeff[13]
                + firstcoeff[8] * othercoeff[0]
                + firstcoeff[22] * othercoeff[16]
                + 2 * firstcoeff[24] * othercoeff[17]
                + 2 * firstcoeff[25] * othercoeff[19]
                + 2 * firstcoeff[30] * othercoeff[26]
                + 2 * firstcoeff[31] * othercoeff[22]
            )
            - (
                firstcoeff[2] * othercoeff[17]
                + firstcoeff[3] * othercoeff[19]
                + firstcoeff[4] * othercoeff[1]
                + firstcoeff[10] * othercoeff[26]
                + firstcoeff[11] * othercoeff[6]
                + firstcoeff[13] * othercoeff[7]
                + 2 * firstcoeff[15] * othercoeff[8]
                + firstcoeff[16] * othercoeff[22]
                + firstcoeff[17] * othercoeff[2]
                + firstcoeff[19] * othercoeff[3]
                + 2 * firstcoeff[21] * othercoeff[4]
                + firstcoeff[26] * othercoeff[10]
                + 2 * firstcoeff[28] * othercoeff[11]
                + 2 * firstcoeff[29] * othercoeff[13]
            ),
            (
                firstcoeff[0] * othercoeff[9]
                + firstcoeff[1] * othercoeff[5]
                + 2 * firstcoeff[5] * othercoeff[21]
                + firstcoeff[6] * othercoeff[12]
                + firstcoeff[7] * othercoeff[14]
                + firstcoeff[9] * othercoeff[0]
                + 2 * firstcoeff[12] * othercoeff[28]
                + 2 * firstcoeff[14] * othercoeff[29]
                + 2 * firstcoeff[18] * othercoeff[24]
                + 2 * firstcoeff[20] * othercoeff[25]
                + firstcoeff[23] * othercoeff[16]
                + 2 * firstcoeff[27] * othercoeff[30]
            )
            - (
                firstcoeff[2] * othercoeff[18]
                + firstcoeff[3] * othercoeff[20]
                + firstcoeff[5] * othercoeff[1]
                + 2 * firstcoeff[9] * othercoeff[15]
                + firstcoeff[10] * othercoeff[27]
                + firstcoeff[12] * othercoeff[6]
                + firstcoeff[14] * othercoeff[7]
                + firstcoeff[16] * othercoeff[23]
                + firstcoeff[18] * othercoeff[2]
                + firstcoeff[20] * othercoeff[3]
                + 2 * firstcoeff[23] * othercoeff[31]
                + firstcoeff[27] * othercoeff[10]
            ),
            (
                firstcoeff[0] * othercoeff[10]
                + firstcoeff[1] * othercoeff[16]
                + firstcoeff[2] * othercoeff[3]
                + firstcoeff[7] * othercoeff[6]
                + 2 * firstcoeff[9] * othercoeff[26]
                + firstcoeff[10] * othercoeff[0]
                + 2 * firstcoeff[12] * othercoeff[13]
                + firstcoeff[16] * othercoeff[1]
                + 2 * firstcoeff[18] * othercoeff[19]
                + 2 * firstcoeff[27] * othercoeff[8]
            )
            - (
                firstcoeff[3] * othercoeff[2]
                + 2 * firstcoeff[5] * othercoeff[22]
                + firstcoeff[6] * othercoeff[7]
                + 2 * firstcoeff[14] * othercoeff[11]
                + 2 * firstcoeff[20] * othercoeff[17]
                + 2 * firstcoeff[23] * othercoeff[4]
            ),
            (
                firstcoeff[0] * othercoeff[11]
                + firstcoeff[1] * othercoeff[17]
                + firstcoeff[2] * othercoeff[4]
                + firstcoeff[7] * othercoeff[26]
                + firstcoeff[8] * othercoeff[6]
                + firstcoeff[10] * othercoeff[13]
                + firstcoeff[11] * othercoeff[0]
                + firstcoeff[16] * othercoeff[19]
                + firstcoeff[17] * othercoeff[1]
                + 2 * firstcoeff[25] * othercoeff[22]
                + firstcoeff[26] * othercoeff[7]
                + 2 * firstcoeff[28] * othercoeff[8]
            )
            - (
                firstcoeff[3] * othercoeff[22]
                + firstcoeff[4] * othercoeff[2]
                + firstcoeff[6] * othercoeff[8]
                + firstcoeff[13] * othercoeff[10]
                + 2 * firstcoeff[15] * othercoeff[11]
                + firstcoeff[19] * othercoeff[16]
                + 2 * firstcoeff[21] * othercoeff[17]
                + firstcoeff[22] * othercoeff[3]
                + 2 * firstcoeff[24] * othercoeff[4]
                + 2 * firstcoeff[29] * othercoeff[26]
                + 2 * firstcoeff[30] * othercoeff[13]
                + 2 * firstcoeff[31] * othercoeff[19]
            ),
            (
                firstcoeff[0] * othercoeff[12]
                + firstcoeff[1] * othercoeff[18]
                + firstcoeff[2] * othercoeff[5]
                + 2 * firstcoeff[5] * othercoeff[24]
                + firstcoeff[7] * othercoeff[27]
                + firstcoeff[9] * othercoeff[6]
                + firstcoeff[10] * othercoeff[14]
                + firstcoeff[12] * othercoeff[0]
                + 2 * firstcoeff[14] * othercoeff[30]
                + firstcoeff[16] * othercoeff[20]
                + firstcoeff[18] * othercoeff[1]
                + 2 * firstcoeff[20] * othercoeff[31]
                + 2 * firstcoeff[23] * othercoeff[25]
                + firstcoeff[27] * othercoeff[7]
            )
            - (
                firstcoeff[3] * othercoeff[23]
                + firstcoeff[5] * othercoeff[2]
                + firstcoeff[6] * othercoeff[9]
                + 2 * firstcoeff[9] * othercoeff[28]
                + 2 * firstcoeff[12] * othercoeff[15]
                + firstcoeff[14] * othercoeff[10]
                + 2 * firstcoeff[18] * othercoeff[21]
                + firstcoeff[20] * othercoeff[16]
                + firstcoeff[23] * othercoeff[3]
                + 2 * firstcoeff[27] * othercoeff[29]
            ),
            (
                firstcoeff[0] * othercoeff[13]
                + firstcoeff[1] * othercoeff[19]
                + firstcoeff[2] * othercoeff[22]
                + firstcoeff[3] * othercoeff[4]
                + firstcoeff[8] * othercoeff[7]
                + firstcoeff[11] * othercoeff[10]
                + firstcoeff[13] * othercoeff[0]
                + firstcoeff[17] * othercoeff[16]
                + firstcoeff[19] * othercoeff[1]
                + firstcoeff[22] * othercoeff[2]
                + 2 * firstcoeff[28] * othercoeff[26]
                + 2 * firstcoeff[29] * othercoeff[8]
                + 2 * firstcoeff[30] * othercoeff[11]
                + 2 * firstcoeff[31] * othercoeff[17]
            )
            - (
                firstcoeff[4] * othercoeff[3]
                + firstcoeff[6] * othercoeff[26]
                + firstcoeff[7] * othercoeff[8]
                + firstcoeff[10] * othercoeff[11]
                + 2 * firstcoeff[15] * othercoeff[13]
                + firstcoeff[16] * othercoeff[17]
                + 2 * firstcoeff[21] * othercoeff[19]
                + 2 * firstcoeff[24] * othercoeff[22]
                + 2 * firstcoeff[25] * othercoeff[4]
                + firstcoeff[26] * othercoeff[6]
            ),
            (
                firstcoeff[0] * othercoeff[14]
                + firstcoeff[1] * othercoeff[20]
                + firstcoeff[2] * othercoeff[23]
                + firstcoeff[3] * othercoeff[5]
                + 2 * firstcoeff[5] * othercoeff[25]
                + firstcoeff[9] * othercoeff[7]
                + firstcoeff[12] * othercoeff[10]
                + firstcoeff[14] * othercoeff[0]
                + firstcoeff[18] * othercoeff[16]
                + firstcoeff[20] * othercoeff[1]
                + firstcoeff[23] * othercoeff[2]
                + 2 * firstcoeff[27] * othercoeff[28]
            )
            - (
                firstcoeff[5] * othercoeff[3]
                + firstcoeff[6] * othercoeff[27]
                + firstcoeff[7] * othercoeff[9]
                + 2 * firstcoeff[9] * othercoeff[29]
                + firstcoeff[10] * othercoeff[12]
                + 2 * firstcoeff[12] * othercoeff[30]
                + 2 * firstcoeff[14] * othercoeff[15]
                + firstcoeff[16] * othercoeff[18]
                + 2 * firstcoeff[18] * othercoeff[31]
                + 2 * firstcoeff[20] * othercoeff[21]
                + 2 * firstcoeff[23] * othercoeff[24]
                + firstcoeff[27] * othercoeff[6]
            ),
            (
                firstcoeff[0] * othercoeff[15]
                + firstcoeff[1] * othercoeff[21]
                + firstcoeff[2] * othercoeff[24]
                + firstcoeff[3] * othercoeff[25]
                + firstcoeff[4] * othercoeff[5]
                + firstcoeff[9] * othercoeff[8]
                + firstcoeff[12] * othercoeff[11]
                + firstcoeff[14] * othercoeff[13]
                + firstcoeff[15] * othercoeff[0]
                + firstcoeff[18] * othercoeff[17]
                + firstcoeff[20] * othercoeff[19]
                + firstcoeff[21] * othercoeff[1]
                + firstcoeff[23] * othercoeff[22]
                + firstcoeff[24] * othercoeff[2]
                + firstcoeff[25] * othercoeff[3]
                + firstcoeff[26] * othercoeff[27]
                + 2 * firstcoeff[28] * othercoeff[28]
                + 2 * firstcoeff[29] * othercoeff[29]
                + 2 * firstcoeff[30] * othercoeff[30]
                + 2 * firstcoeff[31] * othercoeff[31]
            )
            - (
                firstcoeff[5] * othercoeff[4]
                + firstcoeff[6] * othercoeff[28]
                + firstcoeff[7] * othercoeff[29]
                + firstcoeff[8] * othercoeff[9]
                + firstcoeff[10] * othercoeff[30]
                + firstcoeff[11] * othercoeff[12]
                + firstcoeff[13] * othercoeff[14]
                + 2 * firstcoeff[15] * othercoeff[15]
                + firstcoeff[16] * othercoeff[31]
                + firstcoeff[17] * othercoeff[18]
                + firstcoeff[19] * othercoeff[20]
                + 2 * firstcoeff[21] * othercoeff[21]
                + firstcoeff[22] * othercoeff[23]
                + 2 * firstcoeff[24] * othercoeff[24]
                + 2 * firstcoeff[25] * othercoeff[25]
                + firstcoeff[27] * othercoeff[26]
                + firstcoeff[28] * othercoeff[6]
                + firstcoeff[29] * othercoeff[7]
                + firstcoeff[30] * othercoeff[10]
                + firstcoeff[31] * othercoeff[16]
            ),
            (
                firstcoeff[0] * othercoeff[16]
                + firstcoeff[1] * othercoeff[10]
                + firstcoeff[3] * othercoeff[6]
                + 2 * firstcoeff[5] * othercoeff[26]
                + firstcoeff[6] * othercoeff[3]
                + firstcoeff[10] * othercoeff[1]
                + 2 * firstcoeff[12] * othercoeff[19]
                + firstcoeff[16] * othercoeff[0]
                + 2 * firstcoeff[18] * othercoeff[13]
                + 2 * firstcoeff[23] * othercoeff[8]
            )
            - (
                firstcoeff[2] * othercoeff[7]
                + firstcoeff[7] * othercoeff[2]
                + 2 * firstcoeff[9] * othercoeff[22]
                + 2 * firstcoeff[14] * othercoeff[17]
                + 2 * firstcoeff[20] * othercoeff[11]
                + 2 * firstcoeff[27] * othercoeff[4]
            ),
            (
                firstcoeff[0] * othercoeff[17]
                + firstcoeff[1] * othercoeff[11]
                + firstcoeff[3] * othercoeff[26]
                + firstcoeff[4] * othercoeff[6]
                + firstcoeff[6] * othercoeff[4]
                + firstcoeff[10] * othercoeff[19]
                + firstcoeff[11] * othercoeff[1]
                + firstcoeff[16] * othercoeff[13]
                + firstcoeff[17] * othercoeff[0]
                + firstcoeff[22] * othercoeff[7]
                + 2 * firstcoeff[24] * othercoeff[8]
                + 2 * firstcoeff[29] * othercoeff[22]
            )
            - (
                firstcoeff[2] * othercoeff[8]
                + firstcoeff[7] * othercoeff[22]
                + firstcoeff[8] * othercoeff[2]
                + firstcoeff[13] * othercoeff[16]
                + 2 * firstcoeff[15] * othercoeff[17]
                + firstcoeff[19] * othercoeff[10]
                + 2 * firstcoeff[21] * othercoeff[11]
                + 2 * firstcoeff[25] * othercoeff[26]
                + firstcoeff[26] * othercoeff[3]
                + 2 * firstcoeff[28] * othercoeff[4]
                + 2 * firstcoeff[30] * othercoeff[19]
                + 2 * firstcoeff[31] * othercoeff[13]
            ),
            (
                firstcoeff[0] * othercoeff[18]
                + firstcoeff[1] * othercoeff[12]
                + firstcoeff[3] * othercoeff[27]
                + firstcoeff[5] * othercoeff[6]
                + firstcoeff[6] * othercoeff[5]
                + 2 * firstcoeff[9] * othercoeff[24]
                + firstcoeff[10] * othercoeff[20]
                + firstcoeff[12] * othercoeff[1]
                + 2 * firstcoeff[14] * othercoeff[31]
                + firstcoeff[16] * othercoeff[14]
                + firstcoeff[18] * othercoeff[0]
                + 2 * firstcoeff[20] * othercoeff[30]
                + firstcoeff[23] * othercoeff[7]
                + 2 * firstcoeff[27] * othercoeff[25]
            )
            - (
                firstcoeff[2] * othercoeff[9]
                + 2 * firstcoeff[5] * othercoeff[28]
                + firstcoeff[7] * othercoeff[23]
                + firstcoeff[9] * othercoeff[2]
                + 2 * firstcoeff[12] * othercoeff[21]
                + firstcoeff[14] * othercoeff[16]
                + 2 * firstcoeff[18] * othercoeff[15]
                + firstcoeff[20] * othercoeff[10]
                + 2 * firstcoeff[23] * othercoeff[29]
                + firstcoeff[27] * othercoeff[3]
            ),
            (
                firstcoeff[0] * othercoeff[19]
                + firstcoeff[1] * othercoeff[13]
                + firstcoeff[4] * othercoeff[7]
                + firstcoeff[6] * othercoeff[22]
                + firstcoeff[7] * othercoeff[4]
                + firstcoeff[11] * othercoeff[16]
                + firstcoeff[13] * othercoeff[1]
                + firstcoeff[17] * othercoeff[10]
                + firstcoeff[19] * othercoeff[0]
                + 2 * firstcoeff[24] * othercoeff[26]
                + 2 * firstcoeff[25] * othercoeff[8]
                + firstcoeff[26] * othercoeff[2]
                + 2 * firstcoeff[30] * othercoeff[17]
                + 2 * firstcoeff[31] * othercoeff[11]
            )
            - (
                firstcoeff[2] * othercoeff[26]
                + firstcoeff[3] * othercoeff[8]
                + firstcoeff[8] * othercoeff[3]
                + firstcoeff[10] * othercoeff[17]
                + 2 * firstcoeff[15] * othercoeff[19]
                + firstcoeff[16] * othercoeff[11]
                + 2 * firstcoeff[21] * othercoeff[13]
                + firstcoeff[22] * othercoeff[6]
                + 2 * firstcoeff[28] * othercoeff[22]
                + 2 * firstcoeff[29] * othercoeff[4]
            ),
            (
                firstcoeff[0] * othercoeff[20]
                + firstcoeff[1] * othercoeff[14]
                + firstcoeff[5] * othercoeff[7]
                + firstcoeff[6] * othercoeff[23]
                + firstcoeff[7] * othercoeff[5]
                + 2 * firstcoeff[9] * othercoeff[25]
                + firstcoeff[12] * othercoeff[16]
                + firstcoeff[14] * othercoeff[1]
                + firstcoeff[18] * othercoeff[10]
                + firstcoeff[20] * othercoeff[0]
                + 2 * firstcoeff[23] * othercoeff[28]
                + firstcoeff[27] * othercoeff[2]
            )
            - (
                firstcoeff[2] * othercoeff[27]
                + firstcoeff[3] * othercoeff[9]
                + 2 * firstcoeff[5] * othercoeff[29]
                + firstcoeff[9] * othercoeff[3]
                + firstcoeff[10] * othercoeff[18]
                + 2 * firstcoeff[12] * othercoeff[31]
                + 2 * firstcoeff[14] * othercoeff[21]
                + firstcoeff[16] * othercoeff[12]
                + 2 * firstcoeff[18] * othercoeff[30]
                + 2 * firstcoeff[20] * othercoeff[15]
                + firstcoeff[23] * othercoeff[6]
                + 2 * firstcoeff[27] * othercoeff[24]
            ),
            (
                firstcoeff[0] * othercoeff[21]
                + firstcoeff[1] * othercoeff[15]
                + firstcoeff[5] * othercoeff[8]
                + firstcoeff[6] * othercoeff[24]
                + firstcoeff[7] * othercoeff[25]
                + firstcoeff[8] * othercoeff[5]
                + firstcoeff[12] * othercoeff[17]
                + firstcoeff[14] * othercoeff[19]
                + firstcoeff[15] * othercoeff[1]
                + firstcoeff[18] * othercoeff[11]
                + firstcoeff[20] * othercoeff[13]
                + firstcoeff[21] * othercoeff[0]
                + firstcoeff[22] * othercoeff[27]
                + 2 * firstcoeff[24] * othercoeff[28]
                + 2 * firstcoeff[25] * othercoeff[29]
                + firstcoeff[27] * othercoeff[22]
                + firstcoeff[28] * othercoeff[2]
                + firstcoeff[29] * othercoeff[3]
                + 2 * firstcoeff[30] * othercoeff[31]
                + 2 * firstcoeff[31] * othercoeff[30]
            )
            - (
                firstcoeff[2] * othercoeff[28]
                + firstcoeff[3] * othercoeff[29]
                + firstcoeff[4] * othercoeff[9]
                + firstcoeff[9] * othercoeff[4]
                + firstcoeff[10] * othercoeff[31]
                + firstcoeff[11] * othercoeff[18]
                + firstcoeff[13] * othercoeff[20]
                + 2 * firstcoeff[15] * othercoeff[21]
                + firstcoeff[16] * othercoeff[30]
                + firstcoeff[17] * othercoeff[12]
                + firstcoeff[19] * othercoeff[14]
                + 2 * firstcoeff[21] * othercoeff[15]
                + firstcoeff[23] * othercoeff[26]
                + firstcoeff[24] * othercoeff[6]
                + firstcoeff[25] * othercoeff[7]
                + firstcoeff[26] * othercoeff[23]
                + 2 * firstcoeff[28] * othercoeff[24]
                + 2 * firstcoeff[29] * othercoeff[25]
                + firstcoeff[30] * othercoeff[16]
                + firstcoeff[31] * othercoeff[10]
            ),
            (
                firstcoeff[0] * othercoeff[22]
                + firstcoeff[1] * othercoeff[26]
                + firstcoeff[2] * othercoeff[13]
                + firstcoeff[4] * othercoeff[10]
                + firstcoeff[7] * othercoeff[17]
                + firstcoeff[10] * othercoeff[4]
                + firstcoeff[13] * othercoeff[2]
                + firstcoeff[16] * othercoeff[8]
                + firstcoeff[19] * othercoeff[6]
                + firstcoeff[22] * othercoeff[0]
                + 2 * firstcoeff[25] * othercoeff[11]
                + 2 * firstcoeff[28] * othercoeff[19]
            )
            - (
                firstcoeff[3] * othercoeff[11]
                + firstcoeff[6] * othercoeff[19]
                + firstcoeff[8] * othercoeff[16]
                + firstcoeff[11] * othercoeff[3]
                + 2 * firstcoeff[15] * othercoeff[22]
                + firstcoeff[17] * othercoeff[7]
                + 2 * firstcoeff[24] * othercoeff[13]
                + 2 * firstcoeff[21] * othercoeff[26]
                + firstcoeff[26] * othercoeff[1]
                + 2 * firstcoeff[29] * othercoeff[17]
                + 2 * firstcoeff[30] * othercoeff[4]
                + 2 * firstcoeff[31] * othercoeff[8]
            ),
            (
                firstcoeff[0] * othercoeff[23]
                + firstcoeff[1] * othercoeff[27]
                + firstcoeff[2] * othercoeff[14]
                + firstcoeff[5] * othercoeff[10]
                + firstcoeff[7] * othercoeff[18]
                + 2 * firstcoeff[9] * othercoeff[31]
                + firstcoeff[10] * othercoeff[5]
                + firstcoeff[14] * othercoeff[2]
                + 2 * firstcoeff[12] * othercoeff[25]
                + firstcoeff[16] * othercoeff[9]
                + 2 * firstcoeff[18] * othercoeff[29]
                + firstcoeff[20] * othercoeff[6]
                + firstcoeff[23] * othercoeff[0]
                + 2 * firstcoeff[27] * othercoeff[21]
            )
            - (
                firstcoeff[3] * othercoeff[12]
                + 2 * firstcoeff[5] * othercoeff[30]
                + firstcoeff[6] * othercoeff[20]
                + firstcoeff[9] * othercoeff[16]
                + firstcoeff[12] * othercoeff[3]
                + 2 * firstcoeff[14] * othercoeff[24]
                + firstcoeff[18] * othercoeff[7]
                + 2 * firstcoeff[20] * othercoeff[28]
                + 2 * firstcoeff[23] * othercoeff[15]
                + firstcoeff[27] * othercoeff[1]
            ),
            (
                firstcoeff[0] * othercoeff[24]
                + firstcoeff[1] * othercoeff[28]
                + firstcoeff[2] * othercoeff[15]
                + firstcoeff[5] * othercoeff[11]
                + firstcoeff[7] * othercoeff[31]
                + firstcoeff[8] * othercoeff[18]
                + firstcoeff[10] * othercoeff[25]
                + firstcoeff[11] * othercoeff[5]
                + firstcoeff[14] * othercoeff[22]
                + firstcoeff[15] * othercoeff[2]
                + firstcoeff[16] * othercoeff[29]
                + firstcoeff[17] * othercoeff[9]
                + firstcoeff[20] * othercoeff[26]
                + firstcoeff[21] * othercoeff[6]
                + firstcoeff[23] * othercoeff[13]
                + firstcoeff[24] * othercoeff[0]
                + 2 * firstcoeff[25] * othercoeff[30]
                + firstcoeff[26] * othercoeff[20]
                + firstcoeff[29] * othercoeff[16]
                + 2 * firstcoeff[28] * othercoeff[21]
                + firstcoeff[30] * othercoeff[3]
                + firstcoeff[31] * othercoeff[7]
            )
            - (
                firstcoeff[3] * othercoeff[30]
                + firstcoeff[4] * othercoeff[12]
                + firstcoeff[6] * othercoeff[21]
                + firstcoeff[9] * othercoeff[17]
                + firstcoeff[12] * othercoeff[4]
                + firstcoeff[13] * othercoeff[23]
                + 2 * firstcoeff[15] * othercoeff[24]
                + firstcoeff[18] * othercoeff[8]
                + firstcoeff[19] * othercoeff[27]
                + 2 * firstcoeff[21] * othercoeff[28]
                + firstcoeff[22] * othercoeff[14]
                + firstcoeff[25] * othercoeff[10]
                + 2 * firstcoeff[24] * othercoeff[15]
                + firstcoeff[27] * othercoeff[19]
                + firstcoeff[28] * othercoeff[1]
                + 2 * firstcoeff[29] * othercoeff[31]
                + 2 * firstcoeff[30] * othercoeff[25]
                + 2 * firstcoeff[31] * othercoeff[29]
            ),
            (
                firstcoeff[0] * othercoeff[25]
                + firstcoeff[1] * othercoeff[29]
                + firstcoeff[2] * othercoeff[30]
                + firstcoeff[3] * othercoeff[15]
                + firstcoeff[5] * othercoeff[13]
                + firstcoeff[8] * othercoeff[20]
                + firstcoeff[11] * othercoeff[23]
                + firstcoeff[13] * othercoeff[5]
                + firstcoeff[15] * othercoeff[3]
                + firstcoeff[17] * othercoeff[27]
                + firstcoeff[19] * othercoeff[9]
                + firstcoeff[21] * othercoeff[7]
                + firstcoeff[22] * othercoeff[12]
                + firstcoeff[24] * othercoeff[10]
                + firstcoeff[25] * othercoeff[0]
                + firstcoeff[27] * othercoeff[17]
                + 2 * firstcoeff[28] * othercoeff[31]
                + 2 * firstcoeff[29] * othercoeff[21]
                + 2 * firstcoeff[30] * othercoeff[24]
                + 2 * firstcoeff[31] * othercoeff[28]
            )
            - (
                firstcoeff[4] * othercoeff[14]
                + firstcoeff[6] * othercoeff[31]
                + firstcoeff[7] * othercoeff[21]
                + firstcoeff[9] * othercoeff[19]
                + firstcoeff[10] * othercoeff[24]
                + firstcoeff[12] * othercoeff[22]
                + firstcoeff[14] * othercoeff[4]
                + 2 * firstcoeff[15] * othercoeff[25]
                + firstcoeff[16] * othercoeff[28]
                + firstcoeff[18] * othercoeff[26]
                + firstcoeff[20] * othercoeff[8]
                + 2 * firstcoeff[21] * othercoeff[29]
                + firstcoeff[23] * othercoeff[11]
                + 2 * firstcoeff[24] * othercoeff[30]
                + 2 * firstcoeff[25] * othercoeff[15]
                + firstcoeff[26] * othercoeff[18]
                + firstcoeff[28] * othercoeff[16]
                + firstcoeff[29] * othercoeff[1]
                + firstcoeff[30] * othercoeff[2]
                + firstcoeff[31] * othercoeff[6]
            ),
            (
                firstcoeff[0] * othercoeff[26]
                + firstcoeff[1] * othercoeff[22]
                + firstcoeff[3] * othercoeff[17]
                + firstcoeff[6] * othercoeff[13]
                + firstcoeff[8] * othercoeff[10]
                + firstcoeff[10] * othercoeff[8]
                + firstcoeff[13] * othercoeff[6]
                + firstcoeff[16] * othercoeff[4]
                + firstcoeff[19] * othercoeff[2]
                + 2 * firstcoeff[24] * othercoeff[19]
                + firstcoeff[26] * othercoeff[0]
                + 2 * firstcoeff[29] * othercoeff[11]
            )
            - (
                firstcoeff[2] * othercoeff[19]
                + firstcoeff[4] * othercoeff[16]
                + firstcoeff[7] * othercoeff[11]
                + firstcoeff[11] * othercoeff[7]
                + 2 * firstcoeff[15] * othercoeff[26]
                + firstcoeff[17] * othercoeff[3]
                + 2 * firstcoeff[21] * othercoeff[22]
                + firstcoeff[22] * othercoeff[1]
                + 2 * firstcoeff[25] * othercoeff[17]
                + 2 * firstcoeff[28] * othercoeff[13]
                + 2 * firstcoeff[30] * othercoeff[8]
                + 2 * firstcoeff[31] * othercoeff[4]
            ),
            (
                firstcoeff[0] * othercoeff[27]
                + firstcoeff[1] * othercoeff[23]
                + firstcoeff[3] * othercoeff[18]
                + 2 * firstcoeff[5] * othercoeff[31]
                + firstcoeff[6] * othercoeff[14]
                + firstcoeff[9] * othercoeff[10]
                + firstcoeff[10] * othercoeff[9]
                + 2 * firstcoeff[12] * othercoeff[29]
                + firstcoeff[14] * othercoeff[6]
                + firstcoeff[16] * othercoeff[5]
                + 2 * firstcoeff[18] * othercoeff[25]
                + firstcoeff[20] * othercoeff[2]
                + 2 * firstcoeff[23] * othercoeff[21]
                + firstcoeff[27] * othercoeff[0]
            )
            - (
                firstcoeff[2] * othercoeff[20]
                + firstcoeff[7] * othercoeff[12]
                + firstcoeff[5] * othercoeff[16]
                + 2 * firstcoeff[9] * othercoeff[30]
                + firstcoeff[12] * othercoeff[7]
                + 2 * firstcoeff[14] * othercoeff[28]
                + firstcoeff[18] * othercoeff[3]
                + 2 * firstcoeff[20] * othercoeff[24]
                + firstcoeff[23] * othercoeff[1]
                + 2 * firstcoeff[27] * othercoeff[15]
            ),
            (
                firstcoeff[0] * othercoeff[28]
                + firstcoeff[1] * othercoeff[24]
                + firstcoeff[3] * othercoeff[31]
                + firstcoeff[4] * othercoeff[18]
                + firstcoeff[11] * othercoeff[9]
                + firstcoeff[9] * othercoeff[11]
                + firstcoeff[27] * othercoeff[13]
                + firstcoeff[14] * othercoeff[26]
                + firstcoeff[20] * othercoeff[22]
                + firstcoeff[6] * othercoeff[15]
                + firstcoeff[16] * othercoeff[25]
                + firstcoeff[30] * othercoeff[7]
                + 2 * firstcoeff[29] * othercoeff[30]
                + firstcoeff[22] * othercoeff[20]
                + firstcoeff[25] * othercoeff[16]
                + firstcoeff[15] * othercoeff[6]
                + firstcoeff[10] * othercoeff[29]
                + firstcoeff[17] * othercoeff[5]
                + 2 * firstcoeff[24] * othercoeff[21]
                + firstcoeff[21] * othercoeff[2]
                + firstcoeff[28] * othercoeff[0]
                + firstcoeff[31] * othercoeff[3]
            )
            - (
                firstcoeff[2] * othercoeff[21]
                + firstcoeff[5] * othercoeff[17]
                + firstcoeff[7] * othercoeff[30]
                + firstcoeff[8] * othercoeff[12]
                + firstcoeff[12] * othercoeff[8]
                + firstcoeff[13] * othercoeff[27]
                + 2 * firstcoeff[15] * othercoeff[28]
                + firstcoeff[18] * othercoeff[4]
                + firstcoeff[19] * othercoeff[23]
                + 2 * firstcoeff[21] * othercoeff[24]
                + firstcoeff[23] * othercoeff[19]
                + firstcoeff[24] * othercoeff[1]
                + 2 * firstcoeff[25] * othercoeff[31]
                + firstcoeff[26] * othercoeff[14]
                + 2 * firstcoeff[28] * othercoeff[15]
                + firstcoeff[29] * othercoeff[10]
                + 2 * firstcoeff[30] * othercoeff[29]
                + 2 * firstcoeff[31] * othercoeff[25]
            ),
            (
                firstcoeff[0] * othercoeff[29]
                + firstcoeff[1] * othercoeff[25]
                + firstcoeff[4] * othercoeff[20]
                + firstcoeff[6] * othercoeff[30]
                + firstcoeff[7] * othercoeff[15]
                + firstcoeff[9] * othercoeff[13]
                + firstcoeff[11] * othercoeff[27]
                + firstcoeff[13] * othercoeff[9]
                + firstcoeff[15] * othercoeff[7]
                + firstcoeff[17] * othercoeff[23]
                + firstcoeff[19] * othercoeff[5]
                + firstcoeff[21] * othercoeff[3]
                + firstcoeff[23] * othercoeff[17]
                + 2 * firstcoeff[24] * othercoeff[31]
                + 2 * firstcoeff[25] * othercoeff[21]
                + firstcoeff[26] * othercoeff[12]
                + firstcoeff[28] * othercoeff[10]
                + firstcoeff[29] * othercoeff[0]
                + 2 * firstcoeff[30] * othercoeff[28]
                + 2 * firstcoeff[31] * othercoeff[24]
            )
            - (
                firstcoeff[2] * othercoeff[31]
                + firstcoeff[3] * othercoeff[21]
                + firstcoeff[5] * othercoeff[19]
                + firstcoeff[8] * othercoeff[14]
                + firstcoeff[10] * othercoeff[28]
                + firstcoeff[12] * othercoeff[26]
                + firstcoeff[14] * othercoeff[8]
                + 2 * firstcoeff[15] * othercoeff[29]
                + firstcoeff[16] * othercoeff[24]
                + firstcoeff[18] * othercoeff[22]
                + firstcoeff[20] * othercoeff[4]
                + 2 * firstcoeff[21] * othercoeff[25]
                + firstcoeff[22] * othercoeff[18]
                + firstcoeff[24] * othercoeff[16]
                + firstcoeff[25] * othercoeff[1]
                + firstcoeff[27] * othercoeff[11]
                + 2 * firstcoeff[28] * othercoeff[30]
                + 2 * firstcoeff[29] * othercoeff[15]
                + firstcoeff[30] * othercoeff[6]
                + firstcoeff[31] * othercoeff[2]
            ),
            (
                firstcoeff[0] * othercoeff[30]
                + firstcoeff[1] * othercoeff[31]
                + firstcoeff[2] * othercoeff[25]
                + firstcoeff[4] * othercoeff[23]
                + firstcoeff[7] * othercoeff[28]
                + firstcoeff[9] * othercoeff[26]
                + firstcoeff[10] * othercoeff[15]
                + firstcoeff[12] * othercoeff[13]
                + firstcoeff[13] * othercoeff[12]
                + firstcoeff[15] * othercoeff[10]
                + firstcoeff[16] * othercoeff[21]
                + firstcoeff[18] * othercoeff[19]
                + firstcoeff[19] * othercoeff[18]
                + firstcoeff[21] * othercoeff[16]
                + firstcoeff[22] * othercoeff[5]
                + firstcoeff[24] * othercoeff[3]
                + 2 * firstcoeff[25] * othercoeff[24]
                + firstcoeff[27] * othercoeff[8]
                + 2 * firstcoeff[28] * othercoeff[29]
                + firstcoeff[29] * othercoeff[6]
                + firstcoeff[30] * othercoeff[0]
                + firstcoeff[31] * othercoeff[1]
            )
            - (
                firstcoeff[3] * othercoeff[24]
                + firstcoeff[5] * othercoeff[22]
                + firstcoeff[6] * othercoeff[29]
                + firstcoeff[8] * othercoeff[27]
                + firstcoeff[11] * othercoeff[14]
                + firstcoeff[14] * othercoeff[11]
                + 2 * firstcoeff[15] * othercoeff[30]
                + firstcoeff[17] * othercoeff[20]
                + firstcoeff[20] * othercoeff[17]
                + 2 * firstcoeff[21] * othercoeff[31]
                + firstcoeff[23] * othercoeff[4]
                + firstcoeff[25] * othercoeff[2]
                + 2 * firstcoeff[24] * othercoeff[25]
                + firstcoeff[26] * othercoeff[9]
                + firstcoeff[28] * othercoeff[7]
                + 2 * firstcoeff[29] * othercoeff[28]
                + 2 * firstcoeff[30] * othercoeff[15]
                + 2 * firstcoeff[31] * othercoeff[21]
            ),
            (
                firstcoeff[0] * othercoeff[31]
                + firstcoeff[1] * othercoeff[30]
                + firstcoeff[3] * othercoeff[28]
                + firstcoeff[5] * othercoeff[26]
                + firstcoeff[6] * othercoeff[25]
                + firstcoeff[8] * othercoeff[23]
                + firstcoeff[10] * othercoeff[21]
                + firstcoeff[12] * othercoeff[19]
                + firstcoeff[13] * othercoeff[18]
                + firstcoeff[15] * othercoeff[16]
                + firstcoeff[16] * othercoeff[15]
                + firstcoeff[18] * othercoeff[13]
                + firstcoeff[19] * othercoeff[12]
                + firstcoeff[21] * othercoeff[10]
                + firstcoeff[23] * othercoeff[8]
                + 2 * firstcoeff[24] * othercoeff[29]
                + firstcoeff[25] * othercoeff[6]
                + firstcoeff[26] * othercoeff[5]
                + firstcoeff[28] * othercoeff[3]
                + 2 * firstcoeff[29] * othercoeff[24]
                + firstcoeff[30] * othercoeff[1]
                + firstcoeff[31] * othercoeff[0]
            )
            - (
                firstcoeff[2] * othercoeff[29]
                + firstcoeff[4] * othercoeff[27]
                + firstcoeff[7] * othercoeff[24]
                + firstcoeff[9] * othercoeff[22]
                + firstcoeff[11] * othercoeff[20]
                + firstcoeff[14] * othercoeff[17]
                + 2 * firstcoeff[15] * othercoeff[31]
                + firstcoeff[17] * othercoeff[14]
                + firstcoeff[20] * othercoeff[11]
                + 2 * firstcoeff[21] * othercoeff[30]
                + firstcoeff[22] * othercoeff[9]
                + firstcoeff[24] * othercoeff[7]
                + 2 * firstcoeff[25] * othercoeff[28]
                + firstcoeff[27] * othercoeff[4]
                + 2 * firstcoeff[28] * othercoeff[25]
                + firstcoeff[29] * othercoeff[2]
                + 2 * firstcoeff[30] * othercoeff[21]
                + 2 * firstcoeff[31] * othercoeff[15]
            ),
        ]
    )
    return out


@nb.njit(cache=True)
def _invert(objcoeff):  # pragma: no cover
    """Generates inverse of cga_object"""
    out = np.array(
        [
            objcoeff[0] - 2 * objcoeff[15],
            objcoeff[1] - 2 * objcoeff[21],
            objcoeff[2] - 2 * objcoeff[24],
            objcoeff[3] - 2 * objcoeff[25],
            objcoeff[4],
            objcoeff[5],
            -objcoeff[6] + 2 * objcoeff[28],
            -objcoeff[7] + 2 * objcoeff[29],
            -objcoeff[8],
            -objcoeff[9],
            -objcoeff[10] + 2 * objcoeff[30],
            -objcoeff[11],
            -objcoeff[12],
            -objcoeff[13],
            -objcoeff[14],
            -objcoeff[15],
            -objcoeff[16] + 2 * objcoeff[31],
            -objcoeff[17],
            -objcoeff[18],
            -objcoeff[19],
            -objcoeff[20],
            -objcoeff[21],
            -objcoeff[22],
            -objcoeff[23],
            -objcoeff[24],
            -objcoeff[25],
            objcoeff[26],
            objcoeff[27],
            objcoeff[28],
            objcoeff[29],
            objcoeff[30],
            objcoeff[31],
        ]
    )
    return out


@nb.njit(cache=True)
def _inner(firstcoeff, othercoeff):  # pragma: no cover
    """inner product

    Parameters
    ----------
    other : TODO
        TODO
        Returns: TODO

    Returns
    -------

    """
    out = np.array(
        [
            firstcoeff[13] * othercoeff[14]
            + firstcoeff[11] * othercoeff[12]
            + firstcoeff[23] * othercoeff[22]
            + firstcoeff[12] * othercoeff[11]
            + firstcoeff[15] * othercoeff[15]
            + firstcoeff[22] * othercoeff[23]
            - firstcoeff[27] * othercoeff[26]
            + firstcoeff[20] * othercoeff[19]
            - firstcoeff[5] * othercoeff[4]
            - firstcoeff[10] * othercoeff[10]
            - firstcoeff[16] * othercoeff[16]
            + firstcoeff[3] * othercoeff[3]
            + firstcoeff[9] * othercoeff[8]
            - firstcoeff[26] * othercoeff[27]
            + firstcoeff[14] * othercoeff[13]
            - firstcoeff[7] * othercoeff[7]
            + firstcoeff[17] * othercoeff[18]
            + firstcoeff[19] * othercoeff[20]
            + firstcoeff[1] * othercoeff[1]
            + firstcoeff[2] * othercoeff[2]
            + firstcoeff[8] * othercoeff[9]
            - firstcoeff[4] * othercoeff[5]
            - firstcoeff[6] * othercoeff[6]
            + firstcoeff[18] * othercoeff[17],
            -firstcoeff[24] * othercoeff[28]
            + firstcoeff[30] * othercoeff[16]
            + firstcoeff[11] * othercoeff[18]
            + firstcoeff[4] * othercoeff[9]
            - firstcoeff[9] * othercoeff[4]
            - firstcoeff[6] * othercoeff[24]
            + firstcoeff[6] * othercoeff[2]
            + firstcoeff[18] * othercoeff[11]
            - firstcoeff[7] * othercoeff[25]
            + firstcoeff[21] * othercoeff[15]
            + firstcoeff[7] * othercoeff[3]
            - firstcoeff[10] * othercoeff[16]
            + firstcoeff[16] * othercoeff[30]
            - firstcoeff[16] * othercoeff[10]
            + firstcoeff[19] * othercoeff[14]
            + firstcoeff[20] * othercoeff[13]
            + firstcoeff[26] * othercoeff[23]
            + firstcoeff[27] * othercoeff[22]
            - firstcoeff[22] * othercoeff[27]
            - firstcoeff[23] * othercoeff[26]
            - firstcoeff[3] * othercoeff[7]
            - firstcoeff[2] * othercoeff[6]
            + firstcoeff[17] * othercoeff[12]
            + firstcoeff[15] * othercoeff[21]
            - firstcoeff[8] * othercoeff[5]
            + firstcoeff[14] * othercoeff[19]
            + firstcoeff[13] * othercoeff[20]
            + firstcoeff[5] * othercoeff[8]
            + firstcoeff[12] * othercoeff[17]
            + firstcoeff[25] * othercoeff[7]
            + firstcoeff[24] * othercoeff[6]
            - firstcoeff[25] * othercoeff[29]
            - firstcoeff[31] * othercoeff[30]
            - firstcoeff[30] * othercoeff[31]
            + firstcoeff[29] * othercoeff[25]
            + firstcoeff[28] * othercoeff[24],
            -firstcoeff[29] * othercoeff[16]
            + firstcoeff[14] * othercoeff[22]
            + firstcoeff[13] * othercoeff[23]
            - firstcoeff[8] * othercoeff[18]
            + firstcoeff[5] * othercoeff[11]
            - firstcoeff[12] * othercoeff[4]
            - firstcoeff[11] * othercoeff[5]
            + firstcoeff[25] * othercoeff[10]
            + firstcoeff[4] * othercoeff[12]
            - firstcoeff[9] * othercoeff[17]
            + firstcoeff[6] * othercoeff[21]
            - firstcoeff[6] * othercoeff[1]
            - firstcoeff[18] * othercoeff[8]
            + firstcoeff[7] * othercoeff[16]
            - firstcoeff[10] * othercoeff[25]
            + firstcoeff[10] * othercoeff[3]
            - firstcoeff[16] * othercoeff[29]
            + firstcoeff[16] * othercoeff[7]
            - firstcoeff[21] * othercoeff[6]
            + firstcoeff[24] * othercoeff[15]
            + firstcoeff[19] * othercoeff[27]
            + firstcoeff[20] * othercoeff[26]
            - firstcoeff[26] * othercoeff[20]
            - firstcoeff[27] * othercoeff[19]
            + firstcoeff[22] * othercoeff[14]
            + firstcoeff[23] * othercoeff[13]
            - firstcoeff[3] * othercoeff[10]
            - firstcoeff[17] * othercoeff[9]
            + firstcoeff[1] * othercoeff[6]
            + firstcoeff[15] * othercoeff[24]
            - firstcoeff[25] * othercoeff[30]
            + firstcoeff[21] * othercoeff[28]
            + firstcoeff[31] * othercoeff[29]
            + firstcoeff[30] * othercoeff[25]
            + firstcoeff[29] * othercoeff[31]
            - firstcoeff[28] * othercoeff[21],
            firstcoeff[24] * othercoeff[30]
            - firstcoeff[18] * othercoeff[26]
            - firstcoeff[6] * othercoeff[16]
            + firstcoeff[7] * othercoeff[21]
            - firstcoeff[7] * othercoeff[1]
            + firstcoeff[10] * othercoeff[24]
            - firstcoeff[10] * othercoeff[2]
            + firstcoeff[16] * othercoeff[28]
            - firstcoeff[16] * othercoeff[6]
            - firstcoeff[21] * othercoeff[7]
            - firstcoeff[19] * othercoeff[9]
            - firstcoeff[24] * othercoeff[10]
            + firstcoeff[25] * othercoeff[15]
            - firstcoeff[20] * othercoeff[8]
            + firstcoeff[26] * othercoeff[18]
            + firstcoeff[27] * othercoeff[17]
            - firstcoeff[22] * othercoeff[12]
            - firstcoeff[23] * othercoeff[11]
            + firstcoeff[2] * othercoeff[10]
            + firstcoeff[1] * othercoeff[7]
            - firstcoeff[17] * othercoeff[27]
            + firstcoeff[15] * othercoeff[25]
            - firstcoeff[14] * othercoeff[4]
            - firstcoeff[13] * othercoeff[5]
            - firstcoeff[8] * othercoeff[20]
            + firstcoeff[5] * othercoeff[13]
            - firstcoeff[12] * othercoeff[22]
            - firstcoeff[11] * othercoeff[23]
            + firstcoeff[28] * othercoeff[16]
            + firstcoeff[4] * othercoeff[14]
            - firstcoeff[9] * othercoeff[19]
            + firstcoeff[21] * othercoeff[29]
            - firstcoeff[31] * othercoeff[28]
            - firstcoeff[30] * othercoeff[24]
            - firstcoeff[29] * othercoeff[21]
            - firstcoeff[28] * othercoeff[31],
            -2 * firstcoeff[24] * othercoeff[11]
            + firstcoeff[26] * othercoeff[16]
            - 2 * firstcoeff[25] * othercoeff[13]
            - firstcoeff[19] * othercoeff[7]
            - firstcoeff[16] * othercoeff[26]
            - firstcoeff[10] * othercoeff[22]
            - firstcoeff[7] * othercoeff[19]
            - 2 * firstcoeff[21] * othercoeff[8]
            - firstcoeff[22] * othercoeff[10]
            + firstcoeff[3] * othercoeff[13]
            + 2 * firstcoeff[31] * othercoeff[26]
            - firstcoeff[6] * othercoeff[17]
            + firstcoeff[2] * othercoeff[11]
            - firstcoeff[17] * othercoeff[6]
            + firstcoeff[1] * othercoeff[8]
            + 2 * firstcoeff[30] * othercoeff[22]
            - firstcoeff[15] * othercoeff[4]
            - firstcoeff[8] * othercoeff[1]
            - firstcoeff[13] * othercoeff[3]
            + 2 * firstcoeff[29] * othercoeff[19]
            + 2 * firstcoeff[28] * othercoeff[17]
            - firstcoeff[11] * othercoeff[2]
            + firstcoeff[4] * othercoeff[15],
            2 * firstcoeff[23] * othercoeff[30]
            - 2 * firstcoeff[27] * othercoeff[31]
            - firstcoeff[20] * othercoeff[7]
            + 2 * firstcoeff[20] * othercoeff[29]
            - firstcoeff[16] * othercoeff[27]
            - firstcoeff[18] * othercoeff[6]
            - firstcoeff[10] * othercoeff[23]
            - firstcoeff[7] * othercoeff[20]
            + firstcoeff[3] * othercoeff[14]
            - firstcoeff[6] * othercoeff[18]
            + firstcoeff[2] * othercoeff[12]
            + 2 * firstcoeff[18] * othercoeff[28]
            + firstcoeff[1] * othercoeff[9]
            + firstcoeff[15] * othercoeff[5]
            - firstcoeff[14] * othercoeff[3]
            - firstcoeff[9] * othercoeff[1]
            + 2 * firstcoeff[14] * othercoeff[25]
            - firstcoeff[12] * othercoeff[2]
            - firstcoeff[5] * othercoeff[15]
            + 2 * firstcoeff[12] * othercoeff[24]
            - firstcoeff[23] * othercoeff[10]
            + 2 * firstcoeff[9] * othercoeff[21]
            + firstcoeff[27] * othercoeff[16],
            firstcoeff[3] * othercoeff[16]
            - firstcoeff[4] * othercoeff[18]
            - firstcoeff[5] * othercoeff[17]
            + firstcoeff[13] * othercoeff[27]
            + firstcoeff[14] * othercoeff[26]
            + firstcoeff[15] * othercoeff[28]
            + firstcoeff[16] * othercoeff[3]
            - firstcoeff[16] * othercoeff[25]
            - firstcoeff[17] * othercoeff[5]
            - firstcoeff[18] * othercoeff[4]
            - firstcoeff[25] * othercoeff[16]
            + firstcoeff[25] * othercoeff[31]
            + firstcoeff[26] * othercoeff[14]
            + firstcoeff[27] * othercoeff[13]
            + firstcoeff[28] * othercoeff[15]
            + firstcoeff[31] * othercoeff[25],
            -firstcoeff[2] * othercoeff[16]
            - firstcoeff[4] * othercoeff[20]
            - firstcoeff[5] * othercoeff[19]
            - firstcoeff[11] * othercoeff[27]
            - firstcoeff[12] * othercoeff[26]
            + firstcoeff[15] * othercoeff[29]
            - firstcoeff[16] * othercoeff[2]
            + firstcoeff[16] * othercoeff[24]
            - firstcoeff[19] * othercoeff[5]
            - firstcoeff[20] * othercoeff[4]
            + firstcoeff[24] * othercoeff[16]
            - firstcoeff[24] * othercoeff[31]
            - firstcoeff[26] * othercoeff[12]
            - firstcoeff[27] * othercoeff[11]
            + firstcoeff[29] * othercoeff[15]
            - firstcoeff[31] * othercoeff[24],
            firstcoeff[22] * othercoeff[31]
            - firstcoeff[26] * othercoeff[10]
            + firstcoeff[26] * othercoeff[30]
            + firstcoeff[25] * othercoeff[19]
            - firstcoeff[19] * othercoeff[3]
            + firstcoeff[19] * othercoeff[25]
            - firstcoeff[10] * othercoeff[26]
            - firstcoeff[21] * othercoeff[4]
            + firstcoeff[24] * othercoeff[17]
            - firstcoeff[3] * othercoeff[19]
            - firstcoeff[17] * othercoeff[2]
            + firstcoeff[31] * othercoeff[22]
            + firstcoeff[30] * othercoeff[26]
            - firstcoeff[2] * othercoeff[17]
            + firstcoeff[17] * othercoeff[24]
            - firstcoeff[13] * othercoeff[29]
            - firstcoeff[29] * othercoeff[13]
            - firstcoeff[28] * othercoeff[11]
            - firstcoeff[4] * othercoeff[21]
            - firstcoeff[11] * othercoeff[28],
            firstcoeff[27] * othercoeff[30]
            + firstcoeff[25] * othercoeff[20]
            - firstcoeff[20] * othercoeff[3]
            + firstcoeff[20] * othercoeff[25]
            - firstcoeff[10] * othercoeff[27]
            - firstcoeff[18] * othercoeff[2]
            + firstcoeff[18] * othercoeff[24]
            + firstcoeff[21] * othercoeff[5]
            - firstcoeff[23] * othercoeff[31]
            + firstcoeff[24] * othercoeff[18]
            - firstcoeff[3] * othercoeff[20]
            - firstcoeff[31] * othercoeff[23]
            + firstcoeff[30] * othercoeff[27]
            - firstcoeff[2] * othercoeff[18]
            + firstcoeff[14] * othercoeff[29]
            + firstcoeff[29] * othercoeff[14]
            + firstcoeff[28] * othercoeff[12]
            + firstcoeff[5] * othercoeff[21]
            + firstcoeff[12] * othercoeff[28]
            - firstcoeff[27] * othercoeff[10],
            firstcoeff[1] * othercoeff[16]
            - firstcoeff[4] * othercoeff[23]
            - firstcoeff[5] * othercoeff[22]
            + firstcoeff[8] * othercoeff[27]
            + firstcoeff[9] * othercoeff[26]
            + firstcoeff[15] * othercoeff[30]
            + firstcoeff[16] * othercoeff[1]
            - firstcoeff[16] * othercoeff[21]
            - firstcoeff[21] * othercoeff[16]
            + firstcoeff[21] * othercoeff[31]
            - firstcoeff[22] * othercoeff[5]
            - firstcoeff[23] * othercoeff[4]
            + firstcoeff[26] * othercoeff[9]
            + firstcoeff[27] * othercoeff[8]
            + firstcoeff[30] * othercoeff[15]
            + firstcoeff[31] * othercoeff[21],
            -firstcoeff[24] * othercoeff[4]
            + firstcoeff[22] * othercoeff[25]
            + firstcoeff[26] * othercoeff[7]
            - firstcoeff[26] * othercoeff[29]
            + firstcoeff[25] * othercoeff[22]
            - firstcoeff[19] * othercoeff[31]
            - firstcoeff[21] * othercoeff[17]
            + firstcoeff[7] * othercoeff[26]
            - firstcoeff[22] * othercoeff[3]
            - firstcoeff[3] * othercoeff[22]
            + firstcoeff[17] * othercoeff[1]
            - firstcoeff[31] * othercoeff[19]
            - firstcoeff[17] * othercoeff[21]
            + firstcoeff[1] * othercoeff[17]
            - firstcoeff[30] * othercoeff[13]
            - firstcoeff[13] * othercoeff[30]
            - firstcoeff[29] * othercoeff[26]
            + firstcoeff[8] * othercoeff[28]
            + firstcoeff[28] * othercoeff[8]
            - firstcoeff[4] * othercoeff[24],
            firstcoeff[24] * othercoeff[5]
            - firstcoeff[27] * othercoeff[29]
            + firstcoeff[25] * othercoeff[23]
            + firstcoeff[20] * othercoeff[31]
            + firstcoeff[18] * othercoeff[1]
            - firstcoeff[21] * othercoeff[18]
            - firstcoeff[18] * othercoeff[21]
            + firstcoeff[7] * othercoeff[27]
            - firstcoeff[3] * othercoeff[23]
            + firstcoeff[31] * othercoeff[20]
            + firstcoeff[1] * othercoeff[18]
            + firstcoeff[30] * othercoeff[14]
            - firstcoeff[29] * othercoeff[27]
            + firstcoeff[14] * othercoeff[30]
            - firstcoeff[9] * othercoeff[28]
            + firstcoeff[5] * othercoeff[24]
            - firstcoeff[28] * othercoeff[9]
            - firstcoeff[23] * othercoeff[3]
            + firstcoeff[27] * othercoeff[7]
            + firstcoeff[23] * othercoeff[25],
            -firstcoeff[22] * othercoeff[24]
            - firstcoeff[26] * othercoeff[6]
            + firstcoeff[26] * othercoeff[28]
            - firstcoeff[25] * othercoeff[4]
            + firstcoeff[19] * othercoeff[1]
            - firstcoeff[19] * othercoeff[21]
            - firstcoeff[24] * othercoeff[22]
            - firstcoeff[21] * othercoeff[19]
            + firstcoeff[22] * othercoeff[2]
            - firstcoeff[6] * othercoeff[26]
            + firstcoeff[31] * othercoeff[17]
            + firstcoeff[17] * othercoeff[31]
            + firstcoeff[2] * othercoeff[22]
            + firstcoeff[1] * othercoeff[19]
            + firstcoeff[30] * othercoeff[11]
            + firstcoeff[8] * othercoeff[29]
            + firstcoeff[29] * othercoeff[8]
            + firstcoeff[28] * othercoeff[26]
            - firstcoeff[4] * othercoeff[25]
            + firstcoeff[11] * othercoeff[30],
            firstcoeff[27] * othercoeff[28]
            + firstcoeff[20] * othercoeff[1]
            + firstcoeff[25] * othercoeff[5]
            - firstcoeff[20] * othercoeff[21]
            - firstcoeff[24] * othercoeff[23]
            - firstcoeff[21] * othercoeff[20]
            - firstcoeff[6] * othercoeff[27]
            - firstcoeff[18] * othercoeff[31]
            + firstcoeff[2] * othercoeff[23]
            - firstcoeff[31] * othercoeff[18]
            + firstcoeff[1] * othercoeff[20]
            - firstcoeff[30] * othercoeff[12]
            - firstcoeff[29] * othercoeff[9]
            - firstcoeff[9] * othercoeff[29]
            + firstcoeff[28] * othercoeff[27]
            + firstcoeff[5] * othercoeff[25]
            - firstcoeff[12] * othercoeff[30]
            + firstcoeff[23] * othercoeff[2]
            - firstcoeff[23] * othercoeff[24]
            - firstcoeff[27] * othercoeff[6],
            firstcoeff[1] * othercoeff[21]
            + firstcoeff[2] * othercoeff[24]
            + firstcoeff[3] * othercoeff[25]
            - firstcoeff[6] * othercoeff[28]
            - firstcoeff[7] * othercoeff[29]
            - firstcoeff[10] * othercoeff[30]
            - firstcoeff[16] * othercoeff[31]
            + firstcoeff[21] * othercoeff[1]
            - 2 * firstcoeff[21] * othercoeff[21]
            + firstcoeff[24] * othercoeff[2]
            - 2 * firstcoeff[24] * othercoeff[24]
            + firstcoeff[25] * othercoeff[3]
            - 2 * firstcoeff[25] * othercoeff[25]
            - firstcoeff[28] * othercoeff[6]
            + 2 * firstcoeff[28] * othercoeff[28]
            - firstcoeff[29] * othercoeff[7]
            + 2 * firstcoeff[29] * othercoeff[29]
            - firstcoeff[30] * othercoeff[10]
            + 2 * firstcoeff[30] * othercoeff[30]
            - firstcoeff[31] * othercoeff[16]
            + 2 * firstcoeff[31] * othercoeff[31],
            firstcoeff[4] * othercoeff[27]
            + firstcoeff[5] * othercoeff[26]
            + firstcoeff[15] * othercoeff[31]
            - firstcoeff[26] * othercoeff[5]
            - firstcoeff[27] * othercoeff[4]
            + firstcoeff[31] * othercoeff[15],
            firstcoeff[3] * othercoeff[26]
            + firstcoeff[4] * othercoeff[28]
            - firstcoeff[13] * othercoeff[31]
            - firstcoeff[25] * othercoeff[26]
            - firstcoeff[26] * othercoeff[3]
            + firstcoeff[26] * othercoeff[25]
            - firstcoeff[28] * othercoeff[4]
            - firstcoeff[31] * othercoeff[13],
            firstcoeff[3] * othercoeff[27]
            - firstcoeff[5] * othercoeff[28]
            + firstcoeff[14] * othercoeff[31]
            - firstcoeff[25] * othercoeff[27]
            - firstcoeff[27] * othercoeff[3]
            + firstcoeff[27] * othercoeff[25]
            + firstcoeff[28] * othercoeff[5]
            + firstcoeff[31] * othercoeff[14],
            -firstcoeff[2] * othercoeff[26]
            + firstcoeff[4] * othercoeff[29]
            + firstcoeff[11] * othercoeff[31]
            + firstcoeff[24] * othercoeff[26]
            + firstcoeff[26] * othercoeff[2]
            - firstcoeff[26] * othercoeff[24]
            - firstcoeff[29] * othercoeff[4]
            + firstcoeff[31] * othercoeff[11],
            -firstcoeff[2] * othercoeff[27]
            - firstcoeff[5] * othercoeff[29]
            - firstcoeff[12] * othercoeff[31]
            + firstcoeff[24] * othercoeff[27]
            + firstcoeff[27] * othercoeff[2]
            - firstcoeff[27] * othercoeff[24]
            + firstcoeff[29] * othercoeff[5]
            - firstcoeff[31] * othercoeff[12],
            -firstcoeff[2] * othercoeff[28]
            - firstcoeff[3] * othercoeff[29]
            - firstcoeff[10] * othercoeff[31]
            + firstcoeff[24] * othercoeff[28]
            + firstcoeff[25] * othercoeff[29]
            + firstcoeff[28] * othercoeff[2]
            - firstcoeff[28] * othercoeff[24]
            + firstcoeff[29] * othercoeff[3]
            - firstcoeff[29] * othercoeff[25]
            + firstcoeff[30] * othercoeff[31]
            - firstcoeff[31] * othercoeff[10]
            + firstcoeff[31] * othercoeff[30],
            firstcoeff[1] * othercoeff[26]
            + firstcoeff[4] * othercoeff[30]
            - firstcoeff[8] * othercoeff[31]
            - firstcoeff[21] * othercoeff[26]
            - firstcoeff[26] * othercoeff[1]
            + firstcoeff[26] * othercoeff[21]
            - firstcoeff[30] * othercoeff[4]
            - firstcoeff[31] * othercoeff[8],
            firstcoeff[1] * othercoeff[27]
            - firstcoeff[5] * othercoeff[30]
            + firstcoeff[9] * othercoeff[31]
            - firstcoeff[21] * othercoeff[27]
            - firstcoeff[27] * othercoeff[1]
            + firstcoeff[27] * othercoeff[21]
            + firstcoeff[30] * othercoeff[5]
            + firstcoeff[31] * othercoeff[9],
            firstcoeff[1] * othercoeff[28]
            - firstcoeff[3] * othercoeff[30]
            + firstcoeff[7] * othercoeff[31]
            - firstcoeff[21] * othercoeff[28]
            + firstcoeff[25] * othercoeff[30]
            - firstcoeff[28] * othercoeff[1]
            + firstcoeff[28] * othercoeff[21]
            - firstcoeff[29] * othercoeff[31]
            + firstcoeff[30] * othercoeff[3]
            - firstcoeff[30] * othercoeff[25]
            + firstcoeff[31] * othercoeff[7]
            - firstcoeff[31] * othercoeff[29],
            firstcoeff[1] * othercoeff[29]
            + firstcoeff[2] * othercoeff[30]
            - firstcoeff[6] * othercoeff[31]
            - firstcoeff[21] * othercoeff[29]
            - firstcoeff[24] * othercoeff[30]
            + firstcoeff[28] * othercoeff[31]
            - firstcoeff[29] * othercoeff[1]
            + firstcoeff[29] * othercoeff[21]
            - firstcoeff[30] * othercoeff[2]
            + firstcoeff[30] * othercoeff[24]
            - firstcoeff[31] * othercoeff[6]
            + firstcoeff[31] * othercoeff[28],
            -firstcoeff[4] * othercoeff[31] - firstcoeff[31] * othercoeff[4],
            firstcoeff[5] * othercoeff[31] + firstcoeff[31] * othercoeff[5],
            firstcoeff[3] * othercoeff[31]
            - firstcoeff[25] * othercoeff[31]
            + firstcoeff[31] * othercoeff[3]
            - firstcoeff[31] * othercoeff[25],
            -firstcoeff[2] * othercoeff[31]
            + firstcoeff[24] * othercoeff[31]
            - firstcoeff[31] * othercoeff[2]
            + firstcoeff[31] * othercoeff[24],
            firstcoeff[1] * othercoeff[31]
            - firstcoeff[21] * othercoeff[31]
            + firstcoeff[31] * othercoeff[1]
            - firstcoeff[31] * othercoeff[21],
            0,
        ]
    )
    return out


@nb.njit(cache=True)
def _wedge(firstcoeff, othercoeff):  # pragma: no cover
    """outer / wedge product

    Parameters
    ----------
    other : TODO
        TODO
        Returns: TODO

    Returns
    -------

    """
    out = np.array(
        [
            firstcoeff[0] * othercoeff[0]
            + firstcoeff[4] * othercoeff[5]
            - firstcoeff[5] * othercoeff[4]
            - firstcoeff[15] * othercoeff[15],
            firstcoeff[0] * othercoeff[1]
            + firstcoeff[1] * othercoeff[0]
            - firstcoeff[4] * othercoeff[9]
            + firstcoeff[5] * othercoeff[8]
            + firstcoeff[8] * othercoeff[5]
            - firstcoeff[9] * othercoeff[4]
            - firstcoeff[15] * othercoeff[21]
            - firstcoeff[21] * othercoeff[15],
            firstcoeff[0] * othercoeff[2]
            + firstcoeff[2] * othercoeff[0]
            - firstcoeff[4] * othercoeff[12]
            + firstcoeff[5] * othercoeff[11]
            + firstcoeff[11] * othercoeff[5]
            - firstcoeff[12] * othercoeff[4]
            - firstcoeff[15] * othercoeff[24]
            - firstcoeff[24] * othercoeff[15],
            firstcoeff[0] * othercoeff[3]
            + firstcoeff[3] * othercoeff[0]
            - firstcoeff[4] * othercoeff[14]
            + firstcoeff[5] * othercoeff[13]
            + firstcoeff[13] * othercoeff[5]
            - firstcoeff[14] * othercoeff[4]
            - firstcoeff[15] * othercoeff[25]
            - firstcoeff[25] * othercoeff[15],
            firstcoeff[0] * othercoeff[4]
            + firstcoeff[4] * othercoeff[0]
            - firstcoeff[4] * othercoeff[15]
            - firstcoeff[15] * othercoeff[4],
            firstcoeff[0] * othercoeff[5]
            + firstcoeff[5] * othercoeff[0]
            - firstcoeff[5] * othercoeff[15]
            - firstcoeff[15] * othercoeff[5],
            -firstcoeff[2] * othercoeff[1]
            + firstcoeff[1] * othercoeff[2]
            + othercoeff[6] * firstcoeff[0]
            - firstcoeff[12] * othercoeff[8]
            - firstcoeff[8] * othercoeff[12]
            - firstcoeff[28] * othercoeff[15]
            - firstcoeff[15] * othercoeff[28]
            - firstcoeff[21] * othercoeff[24]
            + firstcoeff[11] * othercoeff[9]
            - firstcoeff[5] * othercoeff[17]
            + firstcoeff[4] * othercoeff[18]
            + firstcoeff[6] * othercoeff[0]
            + firstcoeff[24] * othercoeff[21]
            - firstcoeff[18] * othercoeff[4]
            + firstcoeff[17] * othercoeff[5]
            + firstcoeff[9] * othercoeff[11],
            -firstcoeff[3] * othercoeff[1]
            + firstcoeff[1] * othercoeff[3]
            + othercoeff[7] * firstcoeff[0]
            - firstcoeff[29] * othercoeff[15]
            - firstcoeff[8] * othercoeff[14]
            + firstcoeff[13] * othercoeff[9]
            - firstcoeff[15] * othercoeff[29]
            - firstcoeff[21] * othercoeff[25]
            - firstcoeff[5] * othercoeff[19]
            + firstcoeff[4] * othercoeff[20]
            + firstcoeff[7] * othercoeff[0]
            + firstcoeff[25] * othercoeff[21]
            - firstcoeff[20] * othercoeff[4]
            + firstcoeff[19] * othercoeff[5]
            + firstcoeff[9] * othercoeff[13]
            - firstcoeff[14] * othercoeff[8],
            firstcoeff[0] * othercoeff[8]
            + firstcoeff[1] * othercoeff[4]
            - firstcoeff[4] * othercoeff[1]
            + firstcoeff[4] * othercoeff[21]
            + firstcoeff[8] * othercoeff[0]
            - firstcoeff[8] * othercoeff[15]
            - firstcoeff[15] * othercoeff[8]
            - firstcoeff[21] * othercoeff[4],
            firstcoeff[0] * othercoeff[9]
            + firstcoeff[1] * othercoeff[5]
            - firstcoeff[5] * othercoeff[1]
            + firstcoeff[5] * othercoeff[21]
            + firstcoeff[9] * othercoeff[0]
            - firstcoeff[9] * othercoeff[15]
            - firstcoeff[15] * othercoeff[9]
            - firstcoeff[21] * othercoeff[5],
            -firstcoeff[3] * othercoeff[2]
            + firstcoeff[2] * othercoeff[3]
            + othercoeff[10] * firstcoeff[0]
            - firstcoeff[30] * othercoeff[15]
            + firstcoeff[12] * othercoeff[13]
            + firstcoeff[13] * othercoeff[12]
            - firstcoeff[15] * othercoeff[30]
            - firstcoeff[24] * othercoeff[25]
            - firstcoeff[11] * othercoeff[14]
            - firstcoeff[5] * othercoeff[22]
            + firstcoeff[4] * othercoeff[23]
            + firstcoeff[10] * othercoeff[0]
            + firstcoeff[25] * othercoeff[24]
            - firstcoeff[14] * othercoeff[11]
            + firstcoeff[22] * othercoeff[5]
            - firstcoeff[23] * othercoeff[4],
            firstcoeff[0] * othercoeff[11]
            + firstcoeff[2] * othercoeff[4]
            - firstcoeff[4] * othercoeff[2]
            + firstcoeff[4] * othercoeff[24]
            + firstcoeff[11] * othercoeff[0]
            - firstcoeff[11] * othercoeff[15]
            - firstcoeff[15] * othercoeff[11]
            - firstcoeff[24] * othercoeff[4],
            firstcoeff[0] * othercoeff[12]
            + firstcoeff[2] * othercoeff[5]
            - firstcoeff[5] * othercoeff[2]
            + firstcoeff[5] * othercoeff[24]
            + firstcoeff[12] * othercoeff[0]
            - firstcoeff[12] * othercoeff[15]
            - firstcoeff[15] * othercoeff[12]
            - firstcoeff[24] * othercoeff[5],
            firstcoeff[0] * othercoeff[13]
            + firstcoeff[3] * othercoeff[4]
            - firstcoeff[4] * othercoeff[3]
            + firstcoeff[4] * othercoeff[25]
            + firstcoeff[13] * othercoeff[0]
            - firstcoeff[13] * othercoeff[15]
            - firstcoeff[15] * othercoeff[13]
            - firstcoeff[25] * othercoeff[4],
            firstcoeff[0] * othercoeff[14]
            + firstcoeff[3] * othercoeff[5]
            - firstcoeff[5] * othercoeff[3]
            + firstcoeff[5] * othercoeff[25]
            + firstcoeff[14] * othercoeff[0]
            - firstcoeff[14] * othercoeff[15]
            - firstcoeff[15] * othercoeff[14]
            - firstcoeff[25] * othercoeff[5],
            firstcoeff[0] * othercoeff[15]
            + firstcoeff[4] * othercoeff[5]
            - firstcoeff[5] * othercoeff[4]
            + firstcoeff[15] * othercoeff[0]
            - 2 * firstcoeff[15] * othercoeff[15],
            firstcoeff[16] * othercoeff[0]
            + firstcoeff[3] * othercoeff[6]
            - firstcoeff[7] * othercoeff[2]
            + firstcoeff[24] * othercoeff[29]
            + firstcoeff[5] * othercoeff[26]
            + firstcoeff[13] * othercoeff[18]
            + firstcoeff[12] * othercoeff[19]
            - firstcoeff[11] * othercoeff[20]
            - firstcoeff[4] * othercoeff[27]
            - firstcoeff[31] * othercoeff[15]
            - firstcoeff[21] * othercoeff[30]
            - firstcoeff[20] * othercoeff[11]
            + firstcoeff[19] * othercoeff[12]
            - firstcoeff[30] * othercoeff[21]
            + firstcoeff[18] * othercoeff[13]
            + firstcoeff[29] * othercoeff[24]
            - firstcoeff[17] * othercoeff[14]
            - firstcoeff[28] * othercoeff[25]
            - firstcoeff[15] * othercoeff[31]
            - firstcoeff[9] * othercoeff[22]
            + firstcoeff[8] * othercoeff[23]
            - firstcoeff[14] * othercoeff[17]
            - firstcoeff[25] * othercoeff[28]
            - firstcoeff[22] * othercoeff[9]
            + firstcoeff[23] * othercoeff[8]
            + firstcoeff[26] * othercoeff[5]
            - firstcoeff[27] * othercoeff[4]
            + firstcoeff[6] * othercoeff[3]
            + othercoeff[16] * firstcoeff[0]
            + firstcoeff[10] * othercoeff[1]
            + firstcoeff[1] * othercoeff[10]
            - firstcoeff[2] * othercoeff[7],
            firstcoeff[0] * othercoeff[17]
            + firstcoeff[1] * othercoeff[11]
            - firstcoeff[2] * othercoeff[8]
            + firstcoeff[4] * othercoeff[6]
            - firstcoeff[4] * othercoeff[28]
            + firstcoeff[6] * othercoeff[4]
            - firstcoeff[8] * othercoeff[2]
            + firstcoeff[8] * othercoeff[24]
            + firstcoeff[11] * othercoeff[1]
            - firstcoeff[11] * othercoeff[21]
            - firstcoeff[15] * othercoeff[17]
            + firstcoeff[17] * othercoeff[0]
            - firstcoeff[17] * othercoeff[15]
            - firstcoeff[21] * othercoeff[11]
            + firstcoeff[24] * othercoeff[8]
            - firstcoeff[28] * othercoeff[4],
            firstcoeff[0] * othercoeff[18]
            + firstcoeff[1] * othercoeff[12]
            - firstcoeff[2] * othercoeff[9]
            + firstcoeff[5] * othercoeff[6]
            - firstcoeff[5] * othercoeff[28]
            + firstcoeff[6] * othercoeff[5]
            - firstcoeff[9] * othercoeff[2]
            + firstcoeff[9] * othercoeff[24]
            + firstcoeff[12] * othercoeff[1]
            - firstcoeff[12] * othercoeff[21]
            - firstcoeff[15] * othercoeff[18]
            + firstcoeff[18] * othercoeff[0]
            - firstcoeff[18] * othercoeff[15]
            - firstcoeff[21] * othercoeff[12]
            + firstcoeff[24] * othercoeff[9]
            - firstcoeff[28] * othercoeff[5],
            firstcoeff[0] * othercoeff[19]
            + firstcoeff[1] * othercoeff[13]
            - firstcoeff[3] * othercoeff[8]
            + firstcoeff[4] * othercoeff[7]
            - firstcoeff[4] * othercoeff[29]
            + firstcoeff[7] * othercoeff[4]
            - firstcoeff[8] * othercoeff[3]
            + firstcoeff[8] * othercoeff[25]
            + firstcoeff[13] * othercoeff[1]
            - firstcoeff[13] * othercoeff[21]
            - firstcoeff[15] * othercoeff[19]
            + firstcoeff[19] * othercoeff[0]
            - firstcoeff[19] * othercoeff[15]
            - firstcoeff[21] * othercoeff[13]
            + firstcoeff[25] * othercoeff[8]
            - firstcoeff[29] * othercoeff[4],
            firstcoeff[0] * othercoeff[20]
            + firstcoeff[1] * othercoeff[14]
            - firstcoeff[3] * othercoeff[9]
            + firstcoeff[5] * othercoeff[7]
            - firstcoeff[5] * othercoeff[29]
            + firstcoeff[7] * othercoeff[5]
            - firstcoeff[9] * othercoeff[3]
            + firstcoeff[9] * othercoeff[25]
            + firstcoeff[14] * othercoeff[1]
            - firstcoeff[14] * othercoeff[21]
            - firstcoeff[15] * othercoeff[20]
            + firstcoeff[20] * othercoeff[0]
            - firstcoeff[20] * othercoeff[15]
            - firstcoeff[21] * othercoeff[14]
            + firstcoeff[25] * othercoeff[9]
            - firstcoeff[29] * othercoeff[5],
            firstcoeff[0] * othercoeff[21]
            + firstcoeff[1] * othercoeff[15]
            - firstcoeff[4] * othercoeff[9]
            + firstcoeff[5] * othercoeff[8]
            + firstcoeff[8] * othercoeff[5]
            - firstcoeff[9] * othercoeff[4]
            + firstcoeff[15] * othercoeff[1]
            - 2 * firstcoeff[15] * othercoeff[21]
            + firstcoeff[21] * othercoeff[0]
            - 2 * firstcoeff[21] * othercoeff[15],
            firstcoeff[0] * othercoeff[22]
            + firstcoeff[2] * othercoeff[13]
            - firstcoeff[3] * othercoeff[11]
            + firstcoeff[4] * othercoeff[10]
            - firstcoeff[4] * othercoeff[30]
            + firstcoeff[10] * othercoeff[4]
            - firstcoeff[11] * othercoeff[3]
            + firstcoeff[11] * othercoeff[25]
            + firstcoeff[13] * othercoeff[2]
            - firstcoeff[13] * othercoeff[24]
            - firstcoeff[15] * othercoeff[22]
            + firstcoeff[22] * othercoeff[0]
            - firstcoeff[22] * othercoeff[15]
            - firstcoeff[24] * othercoeff[13]
            + firstcoeff[25] * othercoeff[11]
            - firstcoeff[30] * othercoeff[4],
            firstcoeff[0] * othercoeff[23]
            + firstcoeff[2] * othercoeff[14]
            - firstcoeff[3] * othercoeff[12]
            + firstcoeff[5] * othercoeff[10]
            - firstcoeff[5] * othercoeff[30]
            + firstcoeff[10] * othercoeff[5]
            - firstcoeff[12] * othercoeff[3]
            + firstcoeff[12] * othercoeff[25]
            + firstcoeff[14] * othercoeff[2]
            - firstcoeff[14] * othercoeff[24]
            - firstcoeff[15] * othercoeff[23]
            + firstcoeff[23] * othercoeff[0]
            - firstcoeff[23] * othercoeff[15]
            - firstcoeff[24] * othercoeff[14]
            + firstcoeff[25] * othercoeff[12]
            - firstcoeff[30] * othercoeff[5],
            firstcoeff[0] * othercoeff[24]
            + firstcoeff[2] * othercoeff[15]
            - firstcoeff[4] * othercoeff[12]
            + firstcoeff[5] * othercoeff[11]
            + firstcoeff[11] * othercoeff[5]
            - firstcoeff[12] * othercoeff[4]
            + firstcoeff[15] * othercoeff[2]
            - 2 * firstcoeff[15] * othercoeff[24]
            + firstcoeff[24] * othercoeff[0]
            - 2 * firstcoeff[24] * othercoeff[15],
            firstcoeff[0] * othercoeff[25]
            + firstcoeff[3] * othercoeff[15]
            - firstcoeff[4] * othercoeff[14]
            + firstcoeff[5] * othercoeff[13]
            + firstcoeff[13] * othercoeff[5]
            - firstcoeff[14] * othercoeff[4]
            + firstcoeff[15] * othercoeff[3]
            - 2 * firstcoeff[15] * othercoeff[25]
            + firstcoeff[25] * othercoeff[0]
            - 2 * firstcoeff[25] * othercoeff[15],
            firstcoeff[6] * othercoeff[13]
            - firstcoeff[13] * othercoeff[28]
            + firstcoeff[13] * othercoeff[6]
            - firstcoeff[26] * othercoeff[15]
            + firstcoeff[26] * othercoeff[0]
            + othercoeff[26] * firstcoeff[0]
            - firstcoeff[15] * othercoeff[26]
            - firstcoeff[11] * othercoeff[7]
            + firstcoeff[11] * othercoeff[29]
            - firstcoeff[4] * othercoeff[16]
            - firstcoeff[7] * othercoeff[11]
            - firstcoeff[21] * othercoeff[22]
            + firstcoeff[3] * othercoeff[17]
            - firstcoeff[2] * othercoeff[19]
            - firstcoeff[31] * othercoeff[4]
            + firstcoeff[1] * othercoeff[22]
            - firstcoeff[30] * othercoeff[8]
            - firstcoeff[19] * othercoeff[24]
            + firstcoeff[19] * othercoeff[2]
            + firstcoeff[29] * othercoeff[11]
            - firstcoeff[28] * othercoeff[13]
            + firstcoeff[17] * othercoeff[25]
            - firstcoeff[17] * othercoeff[3]
            + firstcoeff[8] * othercoeff[10]
            - firstcoeff[8] * othercoeff[30]
            + firstcoeff[24] * othercoeff[19]
            + firstcoeff[10] * othercoeff[8]
            + firstcoeff[16] * othercoeff[4]
            + firstcoeff[22] * othercoeff[21]
            - firstcoeff[22] * othercoeff[1]
            - firstcoeff[25] * othercoeff[17]
            + firstcoeff[4] * othercoeff[31],
            firstcoeff[20] * othercoeff[2]
            + firstcoeff[6] * othercoeff[14]
            - firstcoeff[12] * othercoeff[7]
            - firstcoeff[5] * othercoeff[16]
            - firstcoeff[14] * othercoeff[28]
            - firstcoeff[15] * othercoeff[27]
            - firstcoeff[27] * othercoeff[15]
            + firstcoeff[27] * othercoeff[0]
            + othercoeff[27] * firstcoeff[0]
            + firstcoeff[12] * othercoeff[29]
            + firstcoeff[5] * othercoeff[31]
            - firstcoeff[21] * othercoeff[23]
            + firstcoeff[3] * othercoeff[18]
            - firstcoeff[2] * othercoeff[20]
            - firstcoeff[31] * othercoeff[5]
            + firstcoeff[1] * othercoeff[23]
            - firstcoeff[30] * othercoeff[9]
            - firstcoeff[20] * othercoeff[24]
            + firstcoeff[29] * othercoeff[12]
            - firstcoeff[18] * othercoeff[3]
            + firstcoeff[18] * othercoeff[25]
            - firstcoeff[28] * othercoeff[14]
            + firstcoeff[9] * othercoeff[10]
            + firstcoeff[14] * othercoeff[6]
            - firstcoeff[9] * othercoeff[30]
            - firstcoeff[7] * othercoeff[12]
            + firstcoeff[24] * othercoeff[20]
            + firstcoeff[10] * othercoeff[9]
            + firstcoeff[16] * othercoeff[5]
            - firstcoeff[25] * othercoeff[18]
            + firstcoeff[23] * othercoeff[21]
            - firstcoeff[23] * othercoeff[1],
            2 * firstcoeff[24] * othercoeff[21]
            + firstcoeff[6] * othercoeff[15]
            - firstcoeff[12] * othercoeff[8]
            - firstcoeff[8] * othercoeff[12]
            - 2 * firstcoeff[28] * othercoeff[15]
            + othercoeff[28] * firstcoeff[0]
            + firstcoeff[28] * othercoeff[0]
            - 2 * firstcoeff[15] * othercoeff[28]
            + firstcoeff[11] * othercoeff[9]
            - firstcoeff[5] * othercoeff[17]
            + firstcoeff[4] * othercoeff[18]
            - firstcoeff[24] * othercoeff[1]
            - 2 * firstcoeff[21] * othercoeff[24]
            - firstcoeff[2] * othercoeff[21]
            + firstcoeff[1] * othercoeff[24]
            - firstcoeff[18] * othercoeff[4]
            + firstcoeff[17] * othercoeff[5]
            + firstcoeff[15] * othercoeff[6]
            + firstcoeff[9] * othercoeff[11]
            + firstcoeff[21] * othercoeff[2],
            -2 * firstcoeff[29] * othercoeff[15]
            - firstcoeff[8] * othercoeff[14]
            + firstcoeff[13] * othercoeff[9]
            + firstcoeff[29] * othercoeff[0]
            + othercoeff[29] * firstcoeff[0]
            - 2 * firstcoeff[15] * othercoeff[29]
            - firstcoeff[5] * othercoeff[19]
            + firstcoeff[4] * othercoeff[20]
            - firstcoeff[3] * othercoeff[21]
            - 2 * firstcoeff[21] * othercoeff[25]
            + firstcoeff[1] * othercoeff[25]
            - firstcoeff[20] * othercoeff[4]
            + firstcoeff[19] * othercoeff[5]
            + firstcoeff[15] * othercoeff[7]
            + firstcoeff[9] * othercoeff[13]
            - firstcoeff[14] * othercoeff[8]
            + firstcoeff[7] * othercoeff[15]
            + 2 * firstcoeff[25] * othercoeff[21]
            - firstcoeff[25] * othercoeff[1]
            + firstcoeff[21] * othercoeff[3],
            -2 * firstcoeff[30] * othercoeff[15]
            + firstcoeff[12] * othercoeff[13]
            + firstcoeff[13] * othercoeff[12]
            + firstcoeff[30] * othercoeff[0]
            + othercoeff[30] * firstcoeff[0]
            - 2 * firstcoeff[15] * othercoeff[30]
            - firstcoeff[11] * othercoeff[14]
            - firstcoeff[5] * othercoeff[22]
            + firstcoeff[4] * othercoeff[23]
            - 2 * firstcoeff[24] * othercoeff[25]
            + firstcoeff[24] * othercoeff[3]
            - firstcoeff[3] * othercoeff[24]
            + firstcoeff[2] * othercoeff[25]
            + firstcoeff[15] * othercoeff[10]
            - firstcoeff[14] * othercoeff[11]
            + firstcoeff[10] * othercoeff[15]
            + firstcoeff[22] * othercoeff[5]
            + 2 * firstcoeff[25] * othercoeff[24]
            - firstcoeff[25] * othercoeff[2]
            - firstcoeff[23] * othercoeff[4],
            othercoeff[31] * firstcoeff[0]
            + firstcoeff[6] * othercoeff[25]
            + 2 * firstcoeff[24] * othercoeff[29]
            + firstcoeff[5] * othercoeff[26]
            + firstcoeff[13] * othercoeff[18]
            + firstcoeff[12] * othercoeff[19]
            - firstcoeff[11] * othercoeff[20]
            - firstcoeff[4] * othercoeff[27]
            - firstcoeff[24] * othercoeff[7]
            - firstcoeff[7] * othercoeff[24]
            + firstcoeff[3] * othercoeff[28]
            - 2 * firstcoeff[31] * othercoeff[15]
            - 2 * firstcoeff[21] * othercoeff[30]
            - firstcoeff[2] * othercoeff[29]
            + firstcoeff[1] * othercoeff[30]
            - firstcoeff[20] * othercoeff[11]
            + firstcoeff[19] * othercoeff[12]
            - 2 * firstcoeff[30] * othercoeff[21]
            + firstcoeff[30] * othercoeff[1]
            + firstcoeff[18] * othercoeff[13]
            + 2 * firstcoeff[29] * othercoeff[24]
            - firstcoeff[29] * othercoeff[2]
            - firstcoeff[17] * othercoeff[14]
            + firstcoeff[28] * othercoeff[3]
            - 2 * firstcoeff[28] * othercoeff[25]
            + firstcoeff[15] * othercoeff[16]
            - 2 * firstcoeff[15] * othercoeff[31]
            - firstcoeff[9] * othercoeff[22]
            + firstcoeff[8] * othercoeff[23]
            - firstcoeff[14] * othercoeff[17]
            + firstcoeff[10] * othercoeff[21]
            + firstcoeff[16] * othercoeff[15]
            - 2 * firstcoeff[25] * othercoeff[28]
            - firstcoeff[22] * othercoeff[9]
            + firstcoeff[25] * othercoeff[6]
            + firstcoeff[23] * othercoeff[8]
            + firstcoeff[26] * othercoeff[5]
            - firstcoeff[27] * othercoeff[4]
            + firstcoeff[21] * othercoeff[10]
            + firstcoeff[31] * othercoeff[0],
        ]
    )
    return out


class cga_object:

    """Element of the CGA with methods for:
    addition, multiplication, subtraction, division by scalars, modolo by scalar
    wedgeproduct, dot product, equality checking, reversion and printing
    """

    dim = 32

    even_indices = np.array([0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 26, 27, 28, 29, 30])
    coeff_names = np.array(
        [
            "",
            "e_1",
            "e_2",
            "e_3",
            "e_i",
            "e_o",
            "e_12",
            "e_13",
            "e_1i",
            "e_1o",
            "e_23",
            "e_2i",
            "e_2o",
            "e_3i",
            "e_3o",
            "e_io",
            "e_123",
            "e_12i",
            "e_12o",
            "e_13i",
            "e_13o",
            "e_1io",
            "e_23i",
            "e_23o",
            "e_2io",
            "e_3io",
            "e_123i",
            "e_123o",
            "e_12io",
            "e_13io",
            "e_23io",
            "e_123io",
        ]
    )

    def __init__(self, gen=[0], even=False):
        """

        Parameters
        ----------
        gen :
             (Default value = [0])
        even :
             (Default value = False)

        Returns
        -------

        """

        if isinstance(gen, cga_object):
            cof = gen.coeff
        elif isinstance(gen, (list, tuple, np.ndarray)):
            if len(gen) >= 33:
                raise ValueError("Maximum array length is 32")
            cof = gen
        else:
            cof = [gen]
        # Version if list is given
        self.coeff = np.zeros(self.dim, dtype=np.cdouble)
        if not even:
            for i in range(len(cof)):
                self.coeff[i] = np.cdouble(cof[i])
        else:
            for i in range(len(self.even_indices)):
                self.coeff[self.even_indices[i]] = np.cdouble(cof[i])

    def __add__(self, other):
        """Addition of cga_object

        Parameters
        ----------
        other : cga_object
            object to add to self
            Returns: addition of cga_object with cga_object

        Returns
        -------

        """
        # This is a rather hacky way of ensuring that addition of different
        # datatypes is commutative. Pending better implementation
        if isinstance(other, cga_object):
            return cga_object(self.coeff + other.coeff)
        else:
            return cga_object([other]) + self

    __radd__ = __add__

    def __sub__(self, other):
        """TODO: Docstring for __sub__.

        Parameters
        ----------
        other : TODO
            TODO

        Returns
        -------
        TODO

        """
        return self + (-other)

    def __rsub__(self, other):
        """TODO: Docstring for __rsub__.

        Parameters
        ----------
        other : TODO
            TODO
            Returns: TODO

        Returns
        -------
        TODO

        """
        return other + (-self)

    def __mul__(self, other):
        if isinstance(other, cga_object):
            return cga_object(_mul(self.coeff, other.coeff))
        else:
            return cga_object(_mul(self.coeff, cga_object(other).coeff))

    __rmul__ = __mul__

    def __truediv__(self, other):
        """division by non cga_objects

        Parameters
        ----------
        other : TODO
            TODO
            Returns: TODO

        Returns
        -------

        """
        if isinstance(other, cga_object):
            raise TypeError("Division of cga_objects not allowed")
        else:
            return cga_object(self.coeff / other)

    def __floordiv__(self, other):
        """division by non cga_objects

        Parameters
        ----------
        other : TODO
            TODO
            Returns: TODO

        Returns
        -------

        """
        if isinstance(other, cga_object):
            raise TypeError("Division of cga_objects not supported")
        elif np.linalg.norm(np.imag(self.coeff)) == 0:
            return cga_object(np.real(self.coeff) // other)
        else:
            raise TypeError(
                "Division of Object with complex coefficients not supported."
            )

    def __xor__(self, other):
        """outer / wedge product

        Parameters
        ----------
        other : TODO
            TODO
            Returns: TODO

        Returns
        -------

        """
        if isinstance(other, cga_object):
            return cga_object(_wedge(self.coeff, other.coeff))
        else:
            return cga_object(_wedge(self.coeff, cga_object(other).coeff))

    def __or__(self, other):
        """inner product

        Parameters
        ----------
        other : TODO
            TODO
            Returns: TODO

        Returns
        -------

        """
        if isinstance(other, cga_object):
            return cga_object(_inner(self.coeff, other.coeff))
        else:
            return cga_object(_inner(self.coeff, cga_object(other).coeff))

    def __pos__(self):
        """TODO: Docstring for __NEG__.

        Parameters
        ----------
        other : TODO
            TODO
            Returns: TODO

        Returns
        -------

        """
        return cga_object(self.coeff)

    def __neg__(self):
        """TODO: Docstring for __NEG__.

        Parameters
        ----------
        other : TODO
            TODO
            Returns: TODO

        Returns
        -------

        """
        return cga_object(-self.coeff)

    def __invert__(self):
        return cga_object(_invert(self.coeff))

    def __eq__(self, other):
        """equality checking

        Parameters
        ----------
        other : TODO
            TODO
            Returns: TODO

        Returns
        -------

        """
        if isinstance(other, cga_object):
            return not (any(self.coeff != other.coeff))
        elif all(self.coeff[1:] == 0):
            return self.coeff[0] == other
        else:
            return False

    def __str__(self):
        """ """
        result = ""
        is_first = True
        for i in range(self.dim):
            if self.coeff[i] == 0:
                continue
            if is_first:
                result += repr(self.coeff[i]) + self.coeff_names[i]
                is_first = False
            else:
                result += " + " + repr(self.coeff[i]) + self.coeff_names[i]
        if result == "":
            return "0"
        return result

    def __repr__(self):
        """ """
        result = ""
        is_first = True
        mul_str = ""
        for i in range(self.dim):
            if i > 0:
                mul_str = "*"
            if self.coeff[i] == 0:
                continue
            if is_first:
                result += repr(self.coeff[i]) + mul_str + self.coeff_names[i]
                is_first = False
            else:
                result += " + " + repr(self.coeff[i]) + mul_str + self.coeff_names[i]
        if result == "":
            return "0"
        return result

    def make_even(self):
        """generates cga_object of even grade with coefficients of self

        Returns: (cga_object) even graded version of self

        Parameters
        ----------

        Returns
        -------


        """
        return cga_object(self.coeff[self.even_indices], even=True)

    def get_even(self):
        """Returns even coefficients of object

        Returns: nd.array

        Parameters
        ----------

        Returns
        -------


        """
        return self.coeff[self.even_indices]
