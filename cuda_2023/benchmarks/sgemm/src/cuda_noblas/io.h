

#ifndef IO_H
#define IO_H

// I/O routines
bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float>&v);
bool writeColMajorMatrixFile(const char *fn, int, int, std::vector<float>&);

void printColMajorMatrix(std::vector<float>& v, int nr_row, int nr_col);
void printRowMajorMatrix(std::vector<float>& v, int nr_row, int nr_col);

#endif
