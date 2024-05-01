/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* I/O routines for reading and writing matrices in column-major
 * layout
 */

#include<fstream>
#include<iostream>
#include<vector>

bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float>&v)
{
  std::cerr << "Opening file:"<< fn << std::endl;
  std::fstream f(fn, std::fstream::in);
  if ( !f.good() ) {
    return false;
  }

  // Read # of rows and cols
  f >> nr_row;
  f >> nr_col;

  float data;
  std::cerr << "Matrix dimension: "<<nr_row<<"x"<<nr_col<<std::endl;
  while (f.good() ) {
    f >> data;
    v.push_back(data);
  }

}

bool writeColMajorMatrixFile(const char *fn, int nr_row, int nr_col, std::vector<float>&v)
{
  std::cerr << "Opening file:"<< fn << " for write." << std::endl;
  std::fstream f(fn, std::fstream::out);
  if ( !f.good() ) {
    return false;
  }

  // Read # of rows and cols
  f << nr_row << " "<<nr_col<<" ";

  float data;
  std::cerr << "Matrix dimension: "<<nr_row<<"x"<<nr_col<<std::endl;
  for (int i = 0; i < v.size(); ++i) {
    f << v[i] << ' ';
  }
  f << "\n";
  return true;

}

void printColMajorMatrix(std::vector<float>& v, int nr_row, int nr_col)
{
    if (v.size() != nr_row*nr_col) {
        std::cerr << "ERROR in printColMajorMatrix: vector size doesn't match the given dimensions" << std::endl;
        std::cerr << "      size=" << v.size() << " rows=" << nr_row << " cols=" << nr_col << std::endl;
        return;
    }

    std::cerr << "Matrix (" << nr_row << " rows x " << nr_col << " cols)" << std::endl;
    for (int i=0; i < nr_row; i++) {
        for (int j=0; j < nr_col; j++) {
            std::cerr << " " << v[j*nr_row + i];
        }
        std::cerr << std::endl;
    }
}

void printRowMajorMatrix(std::vector<float>& v, int nr_row, int nr_col)
{
    if (v.size() != nr_row*nr_col) {
        std::cerr << "ERROR in printRowMajorMatrix: vector size doesn't match the given dimensions" << std::endl;
        std::cerr << "      size=" << v.size() << " rows=" << nr_row << " cols=" << nr_col << std::endl;
        return;
    }

    std::cerr << "Matrix (" << nr_row << " rows x " << nr_col << " cols)" << std::endl;
    for (int i=0; i < nr_row; i++) {
        for (int j=0; j < nr_col; j++) {
            std::cerr << " " << v[j*nr_row + i];
        }
        std::cerr << std::endl;
    }
}

