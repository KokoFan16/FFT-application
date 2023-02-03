#ifndef PROFILER_H_
#define PROFILER_H_

#include <iostream>
#include <string>
#include <map> 
#include <vector>
#include <sstream>
#include <iterator>
#include <mpi.h>
#include <stdio.h>
#include <set>
#include <unordered_set>
#include <math.h>
#include <string.h>
#include <chrono>

extern double logging_cost;
extern int call_count;

class Params {
	public:
		std::string tagloop;
		int nloop;
		std::vector<double> times;
};

// Singleton
class Profiler {
	private:
		std::string filename;
		int write_mode;
	 	void sync_events();
		int nagg;
		void write(char* buf, std::string fp, int n);

		Profiler(); // private constructor to ensure only one instance
		Profiler(const Profiler&) = delete;
		Profiler &operator=(const Profiler&) = delete;

		void instart(std::string outname, double tg, int p, int np, int nts, int wmode, int na=0);
		void iset_context(std::string name, int t);
		void idump();

	public:
		double timegap;
		int ntimestep;
		int curTs;
		int nprocs;
		int rank;
		int dump_count;
		int name_id;
		std::string namespath; // call path of functions
		std::map<std::string, Params> output;
		std::chrono::system_clock::time_point pstart;
		std::set<std::string> noncomEvents;
		std::map<std::string, int> name_encodes;

		static Profiler& getInstance() {
            // The only instance
            // Guaranteed to be lazy initialized
            // Guaranteed that it will be destroyed correctly
			static Profiler instance;
            return instance;
		}

		static void start(std::string outname, double tg, int p, int np, int nts, int wmode, int na) {
			getInstance().instart(outname, tg, p, np, nts, wmode, na);
		}

		static void set_context(std::string name, int t) { getInstance().iset_context(name, t); }
		static void dump() { getInstance().idump(); } // gather info from all the processes
};

#endif /* PROFILER_H */
