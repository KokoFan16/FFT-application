#include "profiler.hpp"
#include "events.hpp"

void Events::constr_help(std::string name) {
	double st = MPI_Wtime();
    auto start = std::chrono::system_clock::now(); // get start time of a event
    start_time = start;

	Profiler& context = Profiler::getInstance();
    // get unique code for each name
    if (comEvent == 1) {
    	if (context.name_encodes[name] == 0){
    		context.name_id += 1;
    		context.name_encodes[name] = context.name_id;
    	}
    }

    std::string nameEncode = (comEvent == 1)? std::to_string(context.name_encodes[name]): name;

	if (context.namespath == "") { context.namespath += nameEncode; } // set name-path as key
	else { context.namespath += ">" + nameEncode; } // concatenate name-path (e.g., main<computation)

    if (comEvent == 0)
    	context.noncomEvents.insert(context.namespath);

    double et = MPI_Wtime();
    logging_cost += (et - st);
    call_count += 1;

}

// constructors with different parameters
Events::Events(std::string n, int ce): name(n), comEvent(ce) { constr_help(n); }

Events::Events(std::string n, int ce, std::string t):
		name(n), comEvent(ce), tags(t) { constr_help(n); }

Events::Events(std::string n, int ce, int loop, int ite):
		name(n), comEvent(ce), is_loop(loop), loop_ite(ite) { constr_help(n); }

Events::Events(std::string n, int ce, std::string t, int loop, int ite):
		name(n), comEvent(ce), tags(t), is_loop(loop), loop_ite(ite) { constr_help(n); }
	
// destructor 
Events::~Events() {

	double st = MPI_Wtime();

    auto end_time = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time-start_time; // calculate duration
    elapsed_time = elapsed_seconds.count();

    Profiler& context = Profiler::getInstance();

    std::string& callpath = context.namespath;
    std::string delimiter = ">";
    size_t found = callpath.rfind(delimiter);
    std::map<std::string, Params>& restore = context.output;

    // set value (time and tag) of each function across all the time-steps
    if (context.curTs == 0 && restore.find(callpath) == restore.end()){
    	restore[callpath].tagloop = tags + '/' + std::to_string(is_loop);
    	restore[callpath].times.push_back(elapsed_time);
        restore[callpath].nloop = 1;
    }
    else {
        if (loop_ite == 0){
        	restore[callpath].times.push_back(elapsed_time);
        	restore[callpath].nloop = 1;
        }
        else {
        	if (is_loop == 2) { restore[callpath].times[context.curTs] += elapsed_time; }
        	else {
        		restore[callpath].times.push_back(elapsed_time);
            	restore[callpath].nloop += 1;
        	}
        }
    }
    callpath = callpath.substr(0, found); // back to last level

    /* check if dump output */
    if (context.timegap > 0 && comEvent == 1) {

		auto pend = std::chrono::system_clock::now();
		std::chrono::duration<double> pelapsed = pend-context.pstart; // calculate duration
		double ptime = pelapsed.count();
		double max_ptime;
		MPI_Allreduce(&ptime, &max_ptime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

		// dump files every n seconds
		if (max_ptime >= context.timegap) {
			context.dump();
			context.pstart = std::chrono::system_clock::now();
			context.dump_count += 1;
		}
    }

    double et = MPI_Wtime();
    logging_cost += (et - st);
}
