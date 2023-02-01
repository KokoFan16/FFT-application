#include "profiler.hpp"
#include "csv_writer.hpp"

static int myceil(int x, int y) { return (x/y + (x % y != 0)); }

Profiler::Profiler(){
	write_mode = 0;
	nagg = 0;
	timegap = 0;
	ntimestep = 0;
	curTs = 0;
	nprocs = 0;
	rank = 0;
	dump_count = 0;
	name_id = 0;
}

void Profiler::iset_context(std::string name, int t) {
	namespath = name;
	curTs = t;
}

void Profiler::instart(std::string outname, double tg, int p, int np, int nts, int wmode, int na) {
	filename = outname;
	write_mode = wmode;
	timegap = tg;
	ntimestep = nts;
	nprocs = np;
	rank = p;
	nagg = na;
	pstart = std::chrono::system_clock::now();
}

void Profiler::sync_events () {

	/* find out local non-common events */
	std::string events;
	for (std::string e: noncomEvents){
		events += e + ":" + output[e].tagloop + ' ';
	}

	// find the max total size of events among all processes and the corresponding rank
	int elocal = events.size();
	int ncomevent_sizes[nprocs];
	MPI_Allgather(&elocal, 1, MPI_INT, ncomevent_sizes, 1,  MPI_INT, MPI_COMM_WORLD);

	/* prepare meta-data for allgatherv */
	int ncomevent_displs[nprocs];
	int total_events_size = 0;
	for (int i = 0; i < nprocs; i++) {
		ncomevent_displs[i] = total_events_size;
		total_events_size += ncomevent_sizes[i];
	}

	/* gather all non-common events */
	char* gather_events = (char*) malloc((total_events_size + 1) * sizeof(char));
	MPI_Allgatherv((char*)events.c_str(), elocal, MPI_CHAR, gather_events, ncomevent_sizes, ncomevent_displs, MPI_CHAR, MPI_COMM_WORLD);
	gather_events[total_events_size] = '\0';

	/* split events from gathered events */
	std::set<std::string> all_noncom_events;
	char *token;
	const char s[2] = " ";
	token = strtok(gather_events, s);
	while( token != NULL ) {
		all_noncom_events.insert(token);
		token = strtok(NULL, s);
	}
	free(gather_events);

	/* add non-common events to all processes */
	for (auto str: all_noncom_events) {
		int pos = str.find(':');
		std::string e = str.substr(0, pos);
		// if the rank doesn't have this event
		if (output.find(e) == output.end()) {
			output[e].tagloop = str.substr(pos+1);
			output[e].times.push_back(0);
//			for (int t = 0; t < curTs+1; t++) {output[e].times.push_back(0);}
		}
	}
}

void Profiler::write(char* buf, std::string fp, int n) {
	std::string gather_message = std::string(buf); // convert it to string

	int found;
	std::vector<std::string> timeGather(output.size());
	std::map<std::string, Params> ::iterator p1; // map pointer
	for (int i = 0; i < n; i++) {
		int pos = gather_message.find(',');
		std::string pmessage = gather_message.substr(0, pos); // message from a process
		int c = 0;
		for (p1 = output.begin(); p1 != output.end(); p1++) {
			found = pmessage.find(" ");
			timeGather[c++] += pmessage.substr(0, found) + "|"; // add to corresponding event
			pmessage.erase(0, found+1);
		}
		gather_message.erase(0, pos+1);
	}

	CSVWrite csv(fp); // open CSV file

	if (rank == 0) {
		std::map<std::string, int> ::iterator p2; // map pointer
		for (p2 = name_encodes.begin(); p2 != name_encodes.end(); p2++) {
			csv << p2->first << std::to_string(p2->second);
		}
		csv << endrow;

		// set CSV file Hearer
		csv << "id" << "tag" << "is_loop" << "times" << endrow;
	}

	std::string loop, tag;
	int c = 0;
	for (p1 = output.begin(); p1 != output.end(); p1++) {
		timeGather[c].pop_back();
		int s = p1->second.tagloop.size();

		loop = p1->second.tagloop[s-1];
		tag = p1->second.tagloop.substr(0, s-2);

		// set CSV file content
		csv << p1->first << tag << loop << timeGather[c] << endrow;
		c++;
	}
}

/// write csv file out
void Profiler::idump() {

	if (rank == 0)
		std::cout << rank << " call dump " << dump_count << std::endl;

//	double st = MPI_Wtime();
	sync_events(); /* Sync events of all processes */
//	double et = MPI_Wtime();

	/* generate the exchanged message */
	std::string message = "";
	std::map<std::string, Params> ::iterator p1; // map pointer
	for (p1 = output.begin(); p1 != output.end(); p1++) {
		int cur_tc = p1->second.times.size();

		for (int i = 0; i < cur_tc; i++){
			char delimiter = '-';
			if (p1->second.nloop > 1) {
				delimiter = '+';
				if (i % p1->second.nloop == p1->second.nloop - 1) delimiter = '-';
			}
			message += std::to_string(p1->second.times[i]) + delimiter;
		}
		message.back() = ' ';
	}

	/* get the max message length */
	int strLen = message.length();
	int max_strlen = 0;
	MPI_Allreduce(&strLen, &max_strlen, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	message.resize(max_strlen, ' '); // padded message to max_strlen
	message.back() = ','; // add comma sat the end of each message

	// gather all the data to one process
	char* gather_buffer = nullptr;

	if (write_mode == 0) { // one process write a file
		long long totalLen = max_strlen * nprocs;
		if (rank == 0)
			gather_buffer = (char*)malloc(totalLen * sizeof(char));

		MPI_Gather((char*)message.c_str(), max_strlen, MPI_CHAR, gather_buffer, max_strlen, MPI_CHAR, 0, MPI_COMM_WORLD);

		if (rank == 0) {
			// create file path
			std::string filePath = filename + "_" + std::to_string(ntimestep) + "_" + std::to_string(nprocs) + ".csv"; //+ to_string(dump_count)
			gather_buffer[totalLen - 1] = '\0';
			write(gather_buffer, filePath, nprocs);
		}
	}
	else if (write_mode == 1) { // two-phase IO
		/// split communicator
		int spliter = myceil(nprocs, nagg);
		int color = rank / spliter;

		MPI_Comm split_comm;
		MPI_Comm_split(MPI_COMM_WORLD, color, rank, &split_comm);

		int split_rank, split_size;
		MPI_Comm_rank(split_comm, &split_rank);
		MPI_Comm_size(split_comm, &split_size);

		long long totalLen = split_size * max_strlen;

		if (split_rank == 0)
			gather_buffer = (char*)malloc(totalLen * sizeof(char));
		MPI_Gather((char*)message.c_str(), max_strlen, MPI_CHAR, gather_buffer, max_strlen, MPI_CHAR, 0, split_comm);

		if (split_rank == 0) {
			std::string filePath = filename + "_" + std::to_string(ntimestep) + "_" + std::to_string(nprocs) + "_" + std::to_string(rank) + ".csv";
			gather_buffer[totalLen - 1] = '\0';
			write(gather_buffer, filePath, split_size);
		}

	}
	else { // file-per-process IO
		std::string filePath = filename + "_" + std::to_string(ntimestep) + "_" + std::to_string(nprocs) + "_" + std::to_string(rank) + ".csv";
		write((char*)message.c_str(), filePath, 1);
	}
	free(gather_buffer);

}
