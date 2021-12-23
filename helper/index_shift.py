import numpy as np
import regex as re

if __name__ == "__main__":
    def index_shift(string, shift):
        """Shift indices in strings of form "a[1][2]*b[3] + c[4]"

        Args:
            string (str): String of definition
            shift (int): index shift value

        Returns: (str) Input string with indices shifted

        """
        sep = []
        # Separate String into Indces as pure ints and rest as strings
        splitted = string.split("[")
        for st in splitted:
            s = st.split("]")
            for x in s:
                sep.append(x)
        for i in range(len(sep)):
            try:
                sep[i] = "["+str(int(sep[i])+shift)+"]"
            except:
                continue
        out = ""
        for x in sep:
            out = out+x

        return out

    mul = '-2*a[28]*b[27] - a[11]*b[11] - a[8]*b[8] + a[2]*b[2] + a[3]*b[3] - a[7]*b[7] + a[4]*b[4] - a[17]*b[17] + 2*a[13]*b[12] + 2*a[24]*b[23] - 2*a[6]*b[5] + 2*a[21]*b[20] + 2*a[19]*b[18] + 2*a[10]*b[9] + 2*a[15]*b[14] + a[1]*b[1],2*a[15]*b[20] - a[4]*b[8] + b[2]*a[1] - a[3]*b[7] + 2*a[6]*b[9] + a[7]*b[3] + 2*a[28]*b[23] + 2*a[21]*b[14] - 2*a[24]*b[27] + 2*a[13]*b[18] - a[17]*b[11] + a[2]*b[1] + 2*a[19]*b[12] - a[11]*b[17] - 2*a[10]*b[5] + a[8]*b[4],2*a[15]*b[23] - a[4]*b[11] + a[3]*b[1] + b[3]*a[1] + 2*a[6]*b[12] - a[7]*b[2] - 2*a[28]*b[20] + 2*a[21]*b[27] + 2*a[24]*b[14] - 2*a[13]*b[5] + a[2]*b[7] + a[17]*b[8] - 2*a[19]*b[9] + a[11]*b[4] + a[8]*b[17] - 2*a[10]*b[18],-2*a[15]*b[5] + a[4]*b[1] + 2*a[6]*b[14] + b[4]*a[1] + a[3]*b[11] + 2*a[28]*b[18] - 2*a[21]*b[9] - 2*a[24]*b[12] - 2*a[13]*b[23] + a[2]*b[8] - a[17]*b[7] - 2*a[19]*b[27] - a[11]*b[3] - a[8]*b[2] - 2*a[10]*b[20] - a[7]*b[17],-a[18]*b[7] + 2*a[31]*b[23] - a[9]*b[2] - a[8]*b[20] + b[5]*a[1] + a[3]*b[12] - 2*a[16]*b[5] - a[7]*b[18] + 2*a[30]*b[20] - a[23]*b[11] + a[27]*b[17] - 2*a[26]*b[14] - a[14]*b[4] - 2*a[25]*b[12] - 2*a[22]*b[9] - a[20]*b[8] + 2*a[29]*b[18] + a[5]*b[1] - a[12]*b[3] + a[2]*b[9] + 2*a[32]*b[27] - a[17]*b[27] - a[11]*b[23] + a[4]*b[14],-a[24]*b[11] + 2*a[24]*b[31] + 2*a[13]*b[25] + a[2]*b[10] - a[17]*b[28] - a[19]*b[7] - a[11]*b[24] + a[28]*b[17] + 2*a[15]*b[26] - a[15]*b[4] + a[4]*b[15] + 2*a[19]*b[29] - a[10]*b[2] + 2*a[10]*b[22] - a[8]*b[21] - 2*a[6]*b[16] + b[6]*a[1] + a[3]*b[13] - a[7]*b[19] - 2*a[28]*b[32] - a[21]*b[8] + 2*a[21]*b[30] - a[13]*b[3] + a[6]*b[1],-2*a[13]*b[9] - 2*a[19]*b[5] + 2*a[10]*b[12] + a[2]*b[3] + a[17]*b[4] + a[11]*b[8] - a[8]*b[11] + 2*a[28]*b[14] + 2*a[15]*b[27] + a[4]*b[17] - 2*a[6]*b[18] + b[7]*a[1] - a[3]*b[2] + a[7]*b[1] + 2*a[21]*b[23] - 2*a[24]*b[20],-a[4]*b[2] - a[3]*b[17] - 2*a[19]*b[23] + 2*a[10]*b[14] - 2*a[21]*b[5] + 2*a[24]*b[18] - 2*a[13]*b[27] - 2*a[28]*b[12] - 2*a[15]*b[9] - 2*a[6]*b[20] + b[8]*a[1] + a[2]*b[4] - a[17]*b[3] - a[11]*b[7] + a[7]*b[11] + a[8]*b[1],-a[18]*b[3] + 2*a[31]*b[27] + a[8]*b[14] - 2*a[16]*b[9] - 2*a[30]*b[14] + a[9]*b[1] + a[7]*b[12] + a[23]*b[17] - a[27]*b[11] - a[14]*b[8] + 2*a[26]*b[20] + 2*a[25]*b[18] - 2*a[22]*b[5] - a[20]*b[4] - 2*a[29]*b[12] - a[5]*b[2] - a[12]*b[7] + a[2]*b[5] + 2*a[32]*b[23] - a[17]*b[23] - a[11]*b[27] - a[4]*b[20] - a[3]*b[18] + b[9]*a[1],2*a[21]*b[26] - a[13]*b[7] + a[24]*b[17] - 2*a[24]*b[32] + 2*a[13]*b[29] + a[2]*b[6] - a[17]*b[24] - a[19]*b[3] + 2*a[19]*b[25] - a[11]*b[28] - a[28]*b[11] - a[6]*b[2] - a[15]*b[8] - a[4]*b[21] - a[3]*b[19] + a[8]*b[15] + a[10]*b[1] - 2*a[10]*b[16] + b[10]*a[1] + 2*a[6]*b[22] + a[7]*b[13] + 2*a[28]*b[31] + 2*a[15]*b[30] - a[21]*b[4],2*a[19]*b[20] - a[4]*b[3] + b[11]*a[1] + a[3]*b[4] + a[2]*b[17] + a[17]*b[2] + a[8]*b[7] + a[11]*b[1] - a[7]*b[8] + 2*a[10]*b[27] - 2*a[21]*b[18] - 2*a[24]*b[5] + 2*a[13]*b[14] - 2*a[6]*b[23] + 2*a[28]*b[9] - 2*a[15]*b[12],-2*a[25]*b[5] - 2*a[22]*b[18] - a[20]*b[17] + 2*a[29]*b[9] - a[5]*b[3] + a[12]*b[1] + a[2]*b[18] - 2*a[32]*b[20] + a[17]*b[20] + a[11]*b[14] + a[18]*b[2] - 2*a[31]*b[14] + a[8]*b[27] + a[9]*b[7] - a[4]*b[23] + a[3]*b[5] + b[12]*a[1] - 2*a[16]*b[12] - 2*a[30]*b[27] - a[7]*b[9] - a[23]*b[4] + a[27]*b[8] - a[14]*b[11] + 2*a[26]*b[23],a[28]*b[8] - a[6]*b[3] - a[15]*b[11] - a[4]*b[24] + a[3]*b[6] + b[13]*a[1] + 2*a[6]*b[25] - a[7]*b[10] - 2*a[28]*b[30] + 2*a[15]*b[31] - a[21]*b[17] + 2*a[21]*b[32] - a[24]*b[4] + 2*a[24]*b[26] + a[13]*b[1] - 2*a[13]*b[16] + a[2]*b[19] + a[19]*b[2] + a[17]*b[21] + a[11]*b[15] - 2*a[19]*b[22] + a[10]*b[7] + a[8]*b[28] - 2*a[10]*b[29],a[3]*b[23] + b[14]*a[1] - 2*a[16]*b[14] - a[7]*b[27] + 2*a[30]*b[9] + a[23]*b[3] - a[27]*b[7] - 2*a[26]*b[5] + a[14]*b[1] - 2*a[25]*b[23] - 2*a[22]*b[20] + a[20]*b[2] + 2*a[29]*b[27] - a[5]*b[4] + a[12]*b[11] + a[2]*b[20] + 2*a[32]*b[18] - a[17]*b[18] - a[11]*b[12] + 2*a[31]*b[12] + a[18]*b[17] - a[8]*b[9] + a[9]*b[8] + a[4]*b[5],a[10]*b[8] - 2*a[19]*b[32] - a[8]*b[10] - 2*a[10]*b[30] + a[4]*b[6] + a[3]*b[24] + b[15]*a[1] + 2*a[6]*b[26] - a[7]*b[28] + 2*a[28]*b[29] + a[15]*b[1] - 2*a[15]*b[16] + a[21]*b[2] - 2*a[21]*b[22] + a[13]*b[11] + a[24]*b[3] - 2*a[24]*b[25] - 2*a[13]*b[31] + a[2]*b[21] + a[19]*b[17] - a[17]*b[19] - a[11]*b[13] - a[28]*b[7] - a[6]*b[4],-a[28]*b[27] + a[13]*b[12] + a[24]*b[23] - a[32]*b[17] - a[31]*b[11] - a[30]*b[8] + a[26]*b[4] + a[25]*b[3] + a[22]*b[2] - a[29]*b[7] + a[16]*b[1] + a[27]*b[28] - 2*a[26]*b[26] - a[14]*b[15] - 2*a[25]*b[25] - 2*a[22]*b[22] - a[20]*b[21] + 2*a[29]*b[29] + a[5]*b[6] + a[4]*b[26] + b[16]*a[1] + a[3]*b[25] - 2*a[16]*b[16] - a[7]*b[29] + 2*a[30]*b[30] - a[12]*b[13] + a[2]*b[22] + 2*a[32]*b[32] - a[17]*b[32] - a[11]*b[31] + 2*a[31]*b[31] - a[18]*b[19] - a[9]*b[10] - a[8]*b[30] - a[23]*b[24] - a[6]*b[5] + a[21]*b[20] + a[19]*b[18] + a[10]*b[9] + a[15]*b[14],-2*a[28]*b[5] - 2*a[15]*b[18] + 2*a[6]*b[27] + a[7]*b[4] + a[2]*b[11] + a[11]*b[2] + a[17]*b[1] - a[8]*b[3] - 2*a[21]*b[12] + 2*a[24]*b[9] + 2*a[13]*b[20] + 2*a[19]*b[14] - 2*a[10]*b[23] + a[4]*b[7] + b[17]*a[1] - a[3]*b[8],2*a[30]*b[23] + a[7]*b[5] + a[23]*b[8] - a[27]*b[4] - 2*a[26]*b[27] - a[14]*b[17] + 2*a[25]*b[9] - 2*a[22]*b[12] - a[20]*b[11] - 2*a[29]*b[5] + a[5]*b[7] + a[12]*b[2] + a[2]*b[12] + a[17]*b[14] - 2*a[32]*b[14] + a[11]*b[20] - 2*a[31]*b[20] + a[18]*b[1] - a[9]*b[3] - a[8]*b[23] + a[4]*b[27] + b[18]*a[1] - a[3]*b[9] - 2*a[16]*b[18],a[7]*b[6] + 2*a[28]*b[26] - 2*a[24]*b[30] + 2*a[15]*b[32] - a[21]*b[11] + 2*a[21]*b[31] + a[13]*b[2] + a[24]*b[8] - 2*a[13]*b[22] + a[2]*b[13] + a[17]*b[15] - a[10]*b[3] + a[11]*b[21] - 2*a[19]*b[16] + a[19]*b[1] + 2*a[10]*b[25] - a[8]*b[24] - a[28]*b[4] + a[6]*b[7] - a[15]*b[17] + a[4]*b[28] - 2*a[6]*b[29] + b[19]*a[1] - a[3]*b[10],2*a[25]*b[27] - 2*a[22]*b[14] - 2*a[29]*b[23] + a[5]*b[8] + a[20]*b[1] + a[12]*b[17] + a[2]*b[14] + 2*a[32]*b[12] - a[17]*b[12] - a[11]*b[18] + a[18]*b[11] + 2*a[31]*b[18] + a[8]*b[5] - a[9]*b[4] - a[4]*b[9] + b[20]*a[1] - a[3]*b[27] - 2*a[16]*b[20] + a[7]*b[23] - 2*a[30]*b[5] - a[23]*b[7] + a[27]*b[3] + 2*a[26]*b[9] + a[14]*b[2],a[28]*b[3] + a[6]*b[8] - 2*a[15]*b[22] + a[15]*b[2] - a[4]*b[10] - 2*a[6]*b[30] + b[21]*a[1] - a[3]*b[28] + a[7]*b[24] + 2*a[24]*b[29] - 2*a[28]*b[25] + a[21]*b[1] - 2*a[21]*b[16] + a[13]*b[17] - a[24]*b[7] - 2*a[13]*b[32] + a[2]*b[15] - a[17]*b[13] + a[19]*b[11] - a[10]*b[4] - a[11]*b[19] - 2*a[19]*b[31] + 2*a[10]*b[26] + a[8]*b[6],a[28]*b[23] + a[21]*b[14] - a[24]*b[27] + a[13]*b[18] + a[15]*b[20] + a[6]*b[9] + a[19]*b[12] - a[10]*b[5] - a[27]*b[24] + 2*a[26]*b[30] - a[14]*b[21] + 2*a[25]*b[29] - 2*a[22]*b[16] - 2*a[29]*b[25] - a[5]*b[10] - a[20]*b[15] + a[2]*b[16] + 2*a[32]*b[31] - a[17]*b[31] - a[11]*b[32] - a[18]*b[13] + 2*a[31]*b[32] + a[8]*b[26] + a[9]*b[6] - a[12]*b[19] - a[4]*b[30] - a[3]*b[29] + b[22]*a[1] - 2*a[16]*b[22] + a[7]*b[25] - 2*a[30]*b[26] + a[23]*b[28] - a[32]*b[11] - a[31]*b[17] + a[16]*b[2] - a[26]*b[8] - a[25]*b[7] + a[22]*b[1] + a[29]*b[3] + a[30]*b[4],a[17]*b[9] + a[11]*b[5] - 2*a[31]*b[5] - a[18]*b[8] + a[8]*b[18] - a[9]*b[17] - a[4]*b[12] + b[23]*a[1] + a[3]*b[14] - 2*a[16]*b[23] - a[7]*b[20] - 2*a[30]*b[18] + a[23]*b[1] - a[27]*b[2] + 2*a[26]*b[12] + a[14]*b[3] - 2*a[25]*b[14] - 2*a[22]*b[27] + a[20]*b[7] + 2*a[29]*b[20] + a[5]*b[11] - a[12]*b[4] + a[2]*b[27] - 2*a[32]*b[9],-a[10]*b[17] + 2*a[19]*b[30] + 2*a[10]*b[32] + a[8]*b[19] - a[28]*b[2] + a[6]*b[11] - 2*a[15]*b[25] + a[15]*b[3] - a[4]*b[13] - 2*a[6]*b[31] + b[24]*a[1] + a[3]*b[15] - a[7]*b[21] + a[24]*b[1] - 2*a[24]*b[16] + 2*a[28]*b[22] + a[21]*b[7] - 2*a[21]*b[29] - a[13]*b[4] + 2*a[13]*b[26] + a[2]*b[28] + a[17]*b[10] - a[19]*b[8] + a[11]*b[6],a[31]*b[4] + a[16]*b[3] + a[30]*b[17] - 2*a[32]*b[30] + a[17]*b[30] + a[11]*b[26] - 2*a[31]*b[26] + a[18]*b[10] + a[8]*b[32] - a[4]*b[31] + a[3]*b[16] + b[25]*a[1] - 2*a[16]*b[25] - 2*a[30]*b[32] - a[7]*b[22] + a[9]*b[19] - a[23]*b[15] + a[27]*b[21] + 2*a[26]*b[31] - a[26]*b[11] + a[25]*b[1] + a[22]*b[7] - a[29]*b[2] + a[32]*b[8] - a[14]*b[24] - 2*a[25]*b[16] - 2*a[22]*b[29] - a[20]*b[28] + 2*a[29]*b[22] - a[5]*b[13] + a[12]*b[6] + a[2]*b[29] + a[15]*b[23] + a[6]*b[12] - a[28]*b[20] + a[21]*b[27] + a[24]*b[14] - a[13]*b[5] - a[19]*b[9] - a[10]*b[18],-a[8]*b[22] + a[4]*b[16] + a[3]*b[31] + b[26]*a[1] - 2*a[16]*b[26] - a[7]*b[32] - 2*a[22]*b[30] + 2*a[29]*b[32] + a[20]*b[10] - a[5]*b[15] + 2*a[30]*b[22] + a[9]*b[21] + a[23]*b[13] - a[27]*b[19] + a[14]*b[6] - 2*a[26]*b[16] - 2*a[25]*b[31] + a[12]*b[24] + a[2]*b[30] + 2*a[32]*b[29] - a[17]*b[29] - a[11]*b[25] + 2*a[31]*b[25] + a[18]*b[28] - a[32]*b[7] - a[31]*b[3] + a[16]*b[4] - a[29]*b[17] - a[30]*b[2] + a[26]*b[1] + a[25]*b[11] + a[22]*b[8] - a[15]*b[5] + a[6]*b[14] + a[28]*b[18] - a[21]*b[9] - a[24]*b[12] - a[13]*b[23] - a[19]*b[27] - a[10]*b[20],a[4]*b[18] - a[3]*b[20] + b[27]*a[1] - 2*a[16]*b[27] + 2*a[30]*b[12] + a[7]*b[14] - a[23]*b[2] + a[27]*b[1] + a[14]*b[7] - 2*a[26]*b[18] + 2*a[25]*b[20] - 2*a[22]*b[23] + a[20]*b[3] - 2*a[29]*b[14] - a[5]*b[17] - a[12]*b[8] + a[2]*b[23] - 2*a[32]*b[5] + a[17]*b[5] + a[11]*b[9] - 2*a[31]*b[9] - a[18]*b[4] - a[8]*b[12] + a[9]*b[11],a[28]*b[1] - 2*a[28]*b[16] - 2*a[15]*b[29] + a[21]*b[3] - 2*a[21]*b[25] - a[13]*b[8] - a[24]*b[2] + 2*a[24]*b[22] + 2*a[13]*b[30] + a[2]*b[24] + a[17]*b[6] - a[19]*b[4] + 2*a[19]*b[26] + a[11]*b[10] + a[10]*b[11] - a[8]*b[13] - 2*a[10]*b[31] - a[6]*b[17] + a[15]*b[7] + a[4]*b[19] - a[3]*b[21] + 2*a[6]*b[32] + b[28]*a[1] + a[7]*b[15],a[4]*b[32] + b[29]*a[1] - a[3]*b[22] - 2*a[16]*b[29] + a[7]*b[16] - a[9]*b[13] + a[16]*b[7] - a[8]*b[31] + a[11]*b[30] + a[18]*b[6] - 2*a[31]*b[30] + a[17]*b[26] + a[31]*b[8] + 2*a[30]*b[31] - a[30]*b[11] + a[23]*b[21] - a[27]*b[15] + a[26]*b[17] - 2*a[26]*b[32] - a[14]*b[28] - a[25]*b[2] + 2*a[25]*b[22] + a[22]*b[3] - 2*a[22]*b[25] - a[20]*b[24] + a[29]*b[1] - 2*a[29]*b[16] + a[5]*b[19] + a[12]*b[10] + a[2]*b[25] + a[32]*b[4] - 2*a[32]*b[26] - a[13]*b[9] - a[19]*b[5] + a[10]*b[12] + a[28]*b[14] + a[15]*b[27] - a[6]*b[18] + a[21]*b[23] - a[24]*b[20],-a[32]*b[3] + 2*a[32]*b[25] - a[17]*b[25] - a[19]*b[23] - a[31]*b[7] - a[11]*b[29] + 2*a[31]*b[29] + a[18]*b[24] + a[10]*b[14] + a[8]*b[16] + a[16]*b[8] - a[26]*b[2] + 2*a[26]*b[22] + a[14]*b[10] - a[25]*b[17] + 2*a[25]*b[32] + a[22]*b[4] - 2*a[22]*b[26] - a[21]*b[5] + a[29]*b[11] + a[20]*b[6] - 2*a[29]*b[31] + a[5]*b[21] + a[24]*b[18] - a[13]*b[27] + a[12]*b[28] + a[2]*b[26] - a[28]*b[12] - a[15]*b[9] - a[4]*b[22] - a[3]*b[32] + b[30]*a[1] - a[6]*b[20] - 2*a[16]*b[30] + a[7]*b[31] + a[30]*b[1] - 2*a[30]*b[16] - a[9]*b[15] - a[23]*b[19] + a[27]*b[13],a[17]*b[22] - a[9]*b[28] + a[11]*b[16] - 2*a[32]*b[22] - 2*a[31]*b[16] + a[2]*b[32] + a[16]*b[11] + a[19]*b[20] + a[32]*b[2] + a[8]*b[29] + a[10]*b[27] + a[31]*b[1] - a[18]*b[21] - a[21]*b[18] - a[29]*b[8] + 2*a[29]*b[30] + a[5]*b[24] - a[24]*b[5] + a[20]*b[19] + a[13]*b[14] - a[12]*b[15] - a[6]*b[23] - 2*a[16]*b[31] - a[7]*b[30] - 2*a[30]*b[29] + a[30]*b[7] + a[28]*b[9] + a[23]*b[6] - a[27]*b[10] - a[26]*b[3] + 2*a[26]*b[25] + a[14]*b[13] + a[25]*b[4] - 2*a[25]*b[26] + a[22]*b[17] - 2*a[22]*b[32] - a[15]*b[12] - a[4]*b[25] + b[31]*a[1] + a[3]*b[26],a[7]*b[26] + 2*a[30]*b[25] - a[30]*b[3] + a[9]*b[24] - a[28]*b[5] - a[23]*b[10] + a[27]*b[6] + a[26]*b[7] - 2*a[26]*b[29] - a[15]*b[18] + a[4]*b[29] - a[3]*b[30] + b[32]*a[1] - 2*a[16]*b[32] + a[6]*b[27] + a[16]*b[17] - a[8]*b[25] - a[18]*b[15] + a[14]*b[19] - a[25]*b[8] + 2*a[25]*b[30] + a[22]*b[11] - a[21]*b[12] - 2*a[22]*b[31] + a[29]*b[4] + a[20]*b[13] - 2*a[29]*b[26] - a[5]*b[28] + a[24]*b[9] + a[13]*b[20] - a[12]*b[21] + a[2]*b[31] + a[32]*b[1] - 2*a[32]*b[16] + a[17]*b[16] + a[11]*b[22] + a[19]*b[14] + a[31]*b[2] - 2*a[31]*b[22] - a[10]*b[23]'


outer = 'a[1]*b[1] + a[5]*b[6] - a[6]*b[5] - a[16]*b[16], a[1]*b[2] + a[2]*b[1] - a[5]*b[10] + a[6]*b[9] + a[9]*b[6] - a[10]*b[5] - a[16]*b[22] - a[22]*b[16], a[1]*b[3] + a[3]*b[1] - a[5]*b[13] + a[6]*b[12] + a[12]*b[6] - a[13]*b[5] - a[16]*b[25] - a[25]*b[16], a[1]*b[4] + a[4]*b[1] - a[5]*b[15] + a[6]*b[14] + a[14]*b[6] - a[15]*b[5] - a[16]*b[26] - a[26]*b[16], a[1]*b[5] + a[5]*b[1] - a[5]*b[16] - a[16]*b[5], a[1]*b[6] + a[6]*b[1] - a[6]*b[16] - a[16]*b[6], -a[3]*b[2] + a[2]*b[3] + b[7]*a[1] - a[13]*b[9] - a[9]*b[13] - a[29]*b[16] - a[16]*b[29] - a[22]*b[25] + a[12]*b[10] - a[6]*b[18] + a[5]*b[19] + a[7]*b[1] + a[25]*b[22] - a[19]*b[5] + a[18]*b[6] + a[10]*b[12], -a[4]*b[2] + a[2]*b[4] + b[8]*a[1] - a[30]*b[16] - a[9]*b[15] + a[14]*b[10] - a[16]*b[30] - a[22]*b[26] - a[6]*b[20] + a[5]*b[21] + a[8]*b[1] + a[26]*b[22] - a[21]*b[5] + a[20]*b[6] + a[10]*b[14] - a[15]*b[9], a[1]*b[9] + a[2]*b[5] - a[5]*b[2] + a[5]*b[22] + a[9]*b[1] - a[9]*b[16] - a[16]*b[9] - a[22]*b[5], a[1]*b[10] + a[2]*b[6] - a[6]*b[2] + a[6]*b[22] + a[10]*b[1] - a[10]*b[16] - a[16]*b[10] - a[22]*b[6], -a[4]*b[3] + a[3]*b[4] + b[11]*a[1] - a[31]*b[16] + a[13]*b[14] + a[14]*b[13] - a[16]*b[31] - a[25]*b[26] - a[12]*b[15] - a[6]*b[23] + a[5]*b[24] + a[11]*b[1] + a[26]*b[25] - a[15]*b[12] + a[23]*b[6] - a[24]*b[5], a[1]*b[12] + a[3]*b[5] - a[5]*b[3] + a[5]*b[25] + a[12]*b[1] - a[12]*b[16] - a[16]*b[12] - a[25]*b[5], a[1]*b[13] + a[3]*b[6] - a[6]*b[3] + a[6]*b[25] + a[13]*b[1] - a[13]*b[16] - a[16]*b[13] - a[25]*b[6], a[1]*b[14] + a[4]*b[5] - a[5]*b[4] + a[5]*b[26] + a[14]*b[1] - a[14]*b[16] - a[16]*b[14] - a[26]*b[5], a[1]*b[15] + a[4]*b[6] - a[6]*b[4] + a[6]*b[26] + a[15]*b[1] - a[15]*b[16] - a[16]*b[15] - a[26]*b[6], a[1]*b[16] + a[5]*b[6] - a[6]*b[5] + a[16]*b[1] - 2*a[16]*b[16], a[17]*b[1] + a[4]*b[7] - a[8]*b[3] + a[25]*b[30] + a[6]*b[27] + a[14]*b[19] + a[13]*b[20] - a[12]*b[21] - a[5]*b[28] - a[32]*b[16] - a[22]*b[31] - a[21]*b[12] + a[20]*b[13] - a[31]*b[22] + a[19]*b[14] + a[30]*b[25] - a[18]*b[15] - a[29]*b[26] - a[16]*b[32] - a[10]*b[23] + a[9]*b[24] - a[15]*b[18] - a[26]*b[29] - a[23]*b[10] + a[24]*b[9] + a[27]*b[6] - a[28]*b[5] + a[7]*b[4] + b[17]*a[1] + a[11]*b[2] + a[2]*b[11] - a[3]*b[8], a[1]*b[18] + a[2]*b[12] - a[3]*b[9] + a[5]*b[7] - a[5]*b[29] + a[7]*b[5] - a[9]*b[3] + a[9]*b[25] + a[12]*b[2] - a[12]*b[22] - a[16]*b[18] + a[18]*b[1] - a[18]*b[16] - a[22]*b[12] + a[25]*b[9] - a[29]*b[5], a[1]*b[19] + a[2]*b[13] - a[3]*b[10] + a[6]*b[7] - a[6]*b[29] + a[7]*b[6] - a[10]*b[3] + a[10]*b[25] + a[13]*b[2] - a[13]*b[22] - a[16]*b[19] + a[19]*b[1] - a[19]*b[16] - a[22]*b[13] + a[25]*b[10] - a[29]*b[6], a[1]*b[20] + a[2]*b[14] - a[4]*b[9] + a[5]*b[8] - a[5]*b[30] + a[8]*b[5] - a[9]*b[4] + a[9]*b[26] + a[14]*b[2] - a[14]*b[22] - a[16]*b[20] + a[20]*b[1] - a[20]*b[16] - a[22]*b[14] + a[26]*b[9] - a[30]*b[5], a[1]*b[21] + a[2]*b[15] - a[4]*b[10] + a[6]*b[8] - a[6]*b[30] + a[8]*b[6] - a[10]*b[4] + a[10]*b[26] + a[15]*b[2] - a[15]*b[22] - a[16]*b[21] + a[21]*b[1] - a[21]*b[16] - a[22]*b[15] + a[26]*b[10] - a[30]*b[6], a[1]*b[22] + a[2]*b[16] - a[5]*b[10] + a[6]*b[9] + a[9]*b[6] - a[10]*b[5] + a[16]*b[2] - 2*a[16]*b[22] + a[22]*b[1] - 2*a[22]*b[16], a[1]*b[23] + a[3]*b[14] - a[4]*b[12] + a[5]*b[11] - a[5]*b[31] + a[11]*b[5] - a[12]*b[4] + a[12]*b[26] + a[14]*b[3] - a[14]*b[25] - a[16]*b[23] + a[23]*b[1] - a[23]*b[16] - a[25]*b[14] + a[26]*b[12] - a[31]*b[5], a[1]*b[24] + a[3]*b[15] - a[4]*b[13] + a[6]*b[11] - a[6]*b[31] + a[11]*b[6] - a[13]*b[4] + a[13]*b[26] + a[15]*b[3] - a[15]*b[25] - a[16]*b[24] + a[24]*b[1] - a[24]*b[16] - a[25]*b[15] + a[26]*b[13] - a[31]*b[6], a[1]*b[25] + a[3]*b[16] - a[5]*b[13] + a[6]*b[12] + a[12]*b[6] - a[13]*b[5] + a[16]*b[3] - 2*a[16]*b[25] + a[25]*b[1] - 2*a[25]*b[16], a[1]*b[26] + a[4]*b[16] - a[5]*b[15] + a[6]*b[14] + a[14]*b[6] - a[15]*b[5] + a[16]*b[4] - 2*a[16]*b[26] + a[26]*b[1] - 2*a[26]*b[16], a[7]*b[14] - a[14]*b[29] + a[14]*b[7] - a[27]*b[16] + a[27]*b[1] + b[27]*a[1] - a[16]*b[27] - a[12]*b[8] + a[12]*b[30] - a[5]*b[17] - a[8]*b[12] - a[22]*b[23] + a[4]*b[18] - a[3]*b[20] - a[32]*b[5] + a[2]*b[23] - a[31]*b[9] - a[20]*b[25] + a[20]*b[3] + a[30]*b[12] - a[29]*b[14] + a[18]*b[26] - a[18]*b[4] + a[9]*b[11] - a[9]*b[31] + a[25]*b[20] + a[11]*b[9] + a[17]*b[5] + a[23]*b[22] - a[23]*b[2] - a[26]*b[18] + a[5]*b[32], a[21]*b[3] + a[7]*b[15] - a[13]*b[8] - a[6]*b[17] - a[15]*b[29] - a[16]*b[28] - a[28]*b[16] + a[28]*b[1] + b[28]*a[1] + a[13]*b[30] + a[6]*b[32] - a[22]*b[24] + a[4]*b[19] - a[3]*b[21] - a[32]*b[6] + a[2]*b[24] - a[31]*b[10] - a[21]*b[25] + a[30]*b[13] - a[19]*b[4] + a[19]*b[26] - a[29]*b[15] + a[10]*b[11] + a[15]*b[7] - a[10]*b[31] - a[8]*b[13] + a[25]*b[21] + a[11]*b[10] + a[17]*b[6] - a[26]*b[19] + a[24]*b[22] - a[24]*b[2], 2*a[25]*b[22] + a[7]*b[16] - a[13]*b[9] - a[9]*b[13] - 2*a[29]*b[16] + b[29]*a[1] + a[29]*b[1] - 2*a[16]*b[29] + a[12]*b[10] - a[6]*b[18] + a[5]*b[19] - a[25]*b[2] - 2*a[22]*b[25] - a[3]*b[22] + a[2]*b[25] - a[19]*b[5] + a[18]*b[6] + a[16]*b[7] + a[10]*b[12] + a[22]*b[3], -2*a[30]*b[16] - a[9]*b[15] + a[14]*b[10] + a[30]*b[1] + b[30]*a[1] - 2*a[16]*b[30] - a[6]*b[20] + a[5]*b[21] - a[4]*b[22] - 2*a[22]*b[26] + a[2]*b[26] - a[21]*b[5] + a[20]*b[6] + a[16]*b[8] + a[10]*b[14] - a[15]*b[9] + a[8]*b[16] + 2*a[26]*b[22] - a[26]*b[2] + a[22]*b[4], -2*a[31]*b[16] + a[13]*b[14] + a[14]*b[13] + a[31]*b[1] + b[31]*a[1] - 2*a[16]*b[31] - a[12]*b[15] - a[6]*b[23] + a[5]*b[24] - 2*a[25]*b[26] + a[25]*b[4] - a[4]*b[25] + a[3]*b[26] + a[16]*b[11] - a[15]*b[12] + a[11]*b[16] + a[23]*b[6] + 2*a[26]*b[25] - a[26]*b[3] - a[24]*b[5], b[32]*a[1] + a[7]*b[26] + 2*a[25]*b[30] + a[6]*b[27] + a[14]*b[19] + a[13]*b[20] - a[12]*b[21] - a[5]*b[28] - a[25]*b[8] - a[8]*b[25] + a[4]*b[29] - 2*a[32]*b[16] - 2*a[22]*b[31] - a[3]*b[30] + a[2]*b[31] - a[21]*b[12] + a[20]*b[13] - 2*a[31]*b[22] + a[31]*b[2] + a[19]*b[14] + 2*a[30]*b[25] - a[30]*b[3] - a[18]*b[15] + a[29]*b[4] - 2*a[29]*b[26] + a[16]*b[17] - 2*a[16]*b[32] - a[10]*b[23] + a[9]*b[24] - a[15]*b[18] + a[11]*b[22] + a[17]*b[16] - 2*a[26]*b[29] - a[23]*b[10] + a[26]*b[7] + a[24]*b[9] + a[27]*b[6] - a[28]*b[5] + a[22]*b[11] + a[32]*b[1]'

inner = 'a[14]*b[15] + a[12]*b[13] + a[24]*b[23] + a[13]*b[12] + a[16]*b[16] + a[23]*b[24] - a[28]*b[27] + a[21]*b[20] - a[6]*b[5] - a[11]*b[11] - a[17]*b[17] + a[4]*b[4] + a[10]*b[9] - a[27]*b[28] + a[15]*b[14] - a[8]*b[8] + a[18]*b[19] + a[20]*b[21] + a[2]*b[2] + a[3]*b[3] + a[9]*b[10] - a[5]*b[6] - a[7]*b[7] + a[19]*b[18],     -a[25]*b[29] + a[31]*b[17] + a[12]*b[19] + a[5]*b[10] - a[10]*b[5] - a[7]*b[25] + a[7]*b[3] + a[19]*b[12] - a[8]*b[26] + a[22]*b[16] + a[8]*b[4] - a[11]*b[17] + a[17]*b[31] - a[17]*b[11] + a[20]*b[15] + a[21]*b[14] + a[27]*b[24] + a[28]*b[23] - a[23]*b[28] - a[24]*b[27] - a[4]*b[8] - a[3]*b[7] + a[18]*b[13] + a[16]*b[22] - a[9]*b[6] + a[15]*b[20] + a[14]*b[21] + a[6]*b[9] + a[13]*b[18] + a[26]*b[8] + a[25]*b[7] - a[26]*b[30] - a[32]*b[31] - a[31]*b[32] + a[30]*b[26] + a[29]*b[25],-a[30]*b[17] + a[15]*b[23] + a[14]*b[24] - a[9]*b[19] + a[6]*b[12] - a[13]*b[5] - a[12]*b[6] + a[26]*b[11] + a[5]*b[13] - a[10]*b[18] + a[7]*b[22] - a[7]*b[2] - a[19]*b[9] + a[8]*b[17] - a[11]*b[26] + a[11]*b[4] - a[17]*b[30] + a[17]*b[8] - a[22]*b[7] + a[25]*b[16] + a[20]*b[28] + a[21]*b[27] - a[27]*b[21] - a[28]*b[20] + a[23]*b[15] + a[24]*b[14] - a[4]*b[11] - a[18]*b[10] + a[2]*b[7] + a[16]*b[25] - a[26]*b[31] + a[22]*b[29] + a[32]*b[30] + a[31]*b[26] + a[30]*b[32] - a[29]*b[22],a[25]*b[31] - a[19]*b[27] - a[7]*b[17] + a[8]*b[22] - a[8]*b[2] + a[11]*b[25] - a[11]*b[3] + a[17]*b[29] - a[17]*b[7] - a[22]*b[8] - a[20]*b[10] - a[25]*b[11] + a[26]*b[16] - a[21]*b[9] + a[27]*b[19] + a[28]*b[18] - a[23]*b[13] - a[24]*b[12] + a[3]*b[11] + a[2]*b[8] - a[18]*b[28] + a[16]*b[26] - a[15]*b[5] - a[14]*b[6] - a[9]*b[21] + a[6]*b[14] - a[13]*b[23] - a[12]*b[24] + a[29]*b[17] + a[5]*b[15] - a[10]*b[20] + a[22]*b[30] - a[32]*b[29] - a[31]*b[25] - a[30]*b[22] - a[29]*b[32],-2*a[25]*b[12] + a[27]*b[17] - 2*a[26]*b[14] - a[20]*b[8] - a[17]*b[27] - a[11]*b[23] - a[8]*b[20] - 2*a[22]*b[9] - a[23]*b[11] + a[4]*b[14] + 2*a[32]*b[27] - a[7]*b[18] + a[3]*b[12] - a[18]*b[7] + a[2]*b[9] + 2*a[31]*b[23] - a[16]*b[5] - a[9]*b[2] - a[14]*b[4] + 2*a[30]*b[20] + 2*a[29]*b[18] - a[12]*b[3] + a[5]*b[16],2*a[24]*b[31] - 2*a[28]*b[32] - a[21]*b[8] + 2*a[21]*b[30] - a[17]*b[28] - a[19]*b[7] - a[11]*b[24] - a[8]*b[21] + a[4]*b[15] - a[7]*b[19] + a[3]*b[13] + 2*a[19]*b[29] + a[2]*b[10] + a[16]*b[6] - a[15]*b[4] - a[10]*b[2] + 2*a[15]*b[26] - a[13]*b[3] - a[6]*b[16] + 2*a[13]*b[25] - a[24]*b[11] + 2*a[10]*b[22] + a[28]*b[17],a[4]*b[17] - a[5]*b[19] - a[6]*b[18] + a[14]*b[28] + a[15]*b[27] + a[16]*b[29] + a[17]*b[4] - a[17]*b[26] - a[18]*b[6] - a[19]*b[5] - a[26]*b[17] + a[26]*b[32] + a[27]*b[15] + a[28]*b[14] + a[29]*b[16] + a[32]*b[26],-a[3]*b[17] - a[5]*b[21] - a[6]*b[20] - a[12]*b[28] - a[13]*b[27] + a[16]*b[30] - a[17]*b[3] + a[17]*b[25] - a[20]*b[6] - a[21]*b[5] + a[25]*b[17] - a[25]*b[32] - a[27]*b[13] - a[28]*b[12] + a[30]*b[16] - a[32]*b[25],a[23]*b[32] - a[27]*b[11] + a[27]*b[31] + a[26]*b[20] - a[20]*b[4] + a[20]*b[26] - a[11]*b[27] - a[22]*b[5] + a[25]*b[18] - a[4]*b[20] - a[18]*b[3] + a[32]*b[23] + a[31]*b[27] - a[3]*b[18] + a[18]*b[25] - a[14]*b[30] - a[30]*b[14] - a[29]*b[12] - a[5]*b[22] - a[12]*b[29],a[28]*b[31] + a[26]*b[21] - a[21]*b[4] + a[21]*b[26] - a[11]*b[28] - a[19]*b[3] + a[19]*b[25] + a[22]*b[6] - a[24]*b[32] + a[25]*b[19] - a[4]*b[21] - a[32]*b[24] + a[31]*b[28] - a[3]*b[19] + a[15]*b[30] + a[30]*b[15] + a[29]*b[13] + a[6]*b[22] + a[13]*b[29] - a[28]*b[11],a[2]*b[17] - a[5]*b[24] - a[6]*b[23] + a[9]*b[28] + a[10]*b[27] + a[16]*b[31] + a[17]*b[2] - a[17]*b[22] - a[22]*b[17] + a[22]*b[32] - a[23]*b[6] - a[24]*b[5] + a[27]*b[10] + a[28]*b[9] + a[31]*b[16] + a[32]*b[22], -a[25]*b[5] + a[23]*b[26] + a[27]*b[8] - a[27]*b[30] + a[26]*b[23] - a[20]*b[32] - a[22]*b[18] + a[8]*b[27] - a[23]*b[4] - a[4]*b[23] + a[18]*b[2] - a[32]*b[20] - a[18]*b[22] + a[2]*b[18] - a[31]*b[14] - a[14]*b[31] - a[30]*b[27] + a[9]*b[29] + a[29]*b[9] - a[5]*b[25], a[25]*b[6] - a[28]*b[30] + a[26]*b[24] + a[21]*b[32] + a[19]*b[2] - a[22]*b[19] - a[19]*b[22] + a[8]*b[28] - a[4]*b[24] + a[32]*b[21] + a[2]*b[19] + a[31]*b[15] - a[30]*b[28] + a[15]*b[31] - a[10]*b[29] + a[6]*b[25] - a[29]*b[10] - a[24]*b[4] + a[28]*b[8] + a[24]*b[26], -a[23]*b[25] - a[27]*b[7] + a[27]*b[29] - a[26]*b[5] + a[20]*b[2] - a[20]*b[22] - a[25]*b[23] - a[22]*b[20] + a[23]*b[3] - a[7]*b[27] + a[32]*b[18] + a[18]*b[32] + a[3]*b[23] + a[2]*b[20] + a[31]*b[12] + a[9]*b[30] + a[30]*b[9] + a[29]*b[27] - a[5]*b[26] + a[12]*b[31], a[28]*b[29] + a[21]*b[2] + a[26]*b[6] - a[21]*b[22] - a[25]*b[24] - a[22]*b[21] - a[7]*b[28] - a[19]*b[32] + a[3]*b[24] - a[32]*b[19] + a[2]*b[21] - a[31]*b[13] - a[30]*b[10] - a[10]*b[30] + a[29]*b[28] + a[6]*b[26] - a[13]*b[31] + a[24]*b[3] - a[24]*b[25] - a[28]*b[7], a[2]*b[22] + a[3]*b[25] + a[4]*b[26] - a[7]*b[29] - a[8]*b[30] - a[11]*b[31] - a[17]*b[32] + a[22]*b[2] - 2*a[22]*b[22] + a[25]*b[3] - 2*a[25]*b[25] + a[26]*b[4] - 2*a[26]*b[26] - a[29]*b[7] + 2*a[29]*b[29] - a[30]*b[8] + 2*a[30]*b[30] - a[31]*b[11] + 2*a[31]*b[31] - a[32]*b[17] + 2*a[32]*b[32], a[5]*b[28] + a[6]*b[27] + a[16]*b[32] - a[27]*b[6] - a[28]*b[5] + a[32]*b[16], a[4]*b[27] + a[5]*b[29] - a[14]*b[32] - a[26]*b[27] - a[27]*b[4] + a[27]*b[26] - a[29]*b[5] - a[32]*b[14], a[4]*b[28] - a[6]*b[29] + a[15]*b[32] - a[26]*b[28] - a[28]*b[4] + a[28]*b[26] + a[29]*b[6] + a[32]*b[15], -a[3]*b[27] + a[5]*b[30] + a[12]*b[32] + a[25]*b[27] + a[27]*b[3] - a[27]*b[25] - a[30]*b[5] + a[32]*b[12], -a[3]*b[28] - a[6]*b[30] - a[13]*b[32] + a[25]*b[28] + a[28]*b[3] - a[28]*b[25] + a[30]*b[6] - a[32]*b[13], -a[3]*b[29] - a[4]*b[30] - a[11]*b[32] + a[25]*b[29] + a[26]*b[30] + a[29]*b[3] - a[29]*b[25] + a[30]*b[4] - a[30]*b[26] + a[31]*b[32] - a[32]*b[11] + a[32]*b[31], a[2]*b[27] + a[5]*b[31] - a[9]*b[32] - a[22]*b[27] - a[27]*b[2] + a[27]*b[22] - a[31]*b[5] - a[32]*b[9], a[2]*b[28] - a[6]*b[31] + a[10]*b[32] - a[22]*b[28] - a[28]*b[2] + a[28]*b[22] + a[31]*b[6] + a[32]*b[10], a[2]*b[29] - a[4]*b[31] + a[8]*b[32] - a[22]*b[29] + a[26]*b[31] - a[29]*b[2] + a[29]*b[22] - a[30]*b[32] + a[31]*b[4] - a[31]*b[26] + a[32]*b[8] - a[32]*b[30], a[2]*b[30] + a[3]*b[31] - a[7]*b[32] - a[22]*b[30] - a[25]*b[31] + a[29]*b[32] - a[30]*b[2] + a[30]*b[22] - a[31]*b[3] + a[31]*b[25] - a[32]*b[7] + a[32]*b[29], -a[5]*b[32] - a[32]*b[5], a[6]*b[32] + a[32]*b[6], a[4]*b[32] - a[26]*b[32] + a[32]*b[4] - a[32]*b[26], -a[3]*b[32] + a[25]*b[32] - a[32]*b[3] + a[32]*b[25], a[2]*b[32] - a[22]*b[32] + a[32]*b[2] - a[32]*b[22], 0]'

reverse = 'a[1] - 2*a[16], a[2] - 2*a[22], a[3] - 2*a[25], a[4] - 2*a[26], a[5], a[6], -a[7] + 2*a[29], -a[8] + 2*a[30], -a[9], -a[10], -a[11] + 2*a[31], -a[12], -a[13], -a[14], -a[15], -a[16], -a[17] + 2*a[32], -a[18], -a[19], -a[20], -a[21], -a[22], -a[23], -a[24], -a[25], -a[26], a[27], a[28], a[29], a[30], a[31], a[32]'

