/*
 * csv_write.cpp
 *
 *      Author: kokofan
 */

#include "csv_writer.hpp"

CSVWrite& CSVWrite::write (const std::string & val){
	if (!begin){ fs << delimiter; }
	else { begin = false; }
	fs << val;
	return *this;
}

CSVWrite::CSVWrite(const std::string filename, const std::string deli): fs(), delimiter(deli), begin(true) {
	fs.exceptions(std::ios::failbit | std::ios::badbit);
	fs.open(filename); // open file
}

CSVWrite::~CSVWrite() { fs.flush(); fs.close(); } // flush and close file

void CSVWrite::endrow() { fs << std::endl; begin = true; } // end of each line

CSVWrite& CSVWrite::operator << ( CSVWrite& (* val)(CSVWrite&)) { return val(*this); } // overwrite operator <<

CSVWrite& CSVWrite::operator << (const std::string & val) { return write(val); } // write string






