int n_size = 1024;

void pre_processing(TString infile = "../peak_classification/counting_test.root", TString outfile = "dataset_train.txt", int start = 0, int n = -1) {
    TFile* f = new TFile(infile);
    TTree* t = (TTree*)f->Get("signal");

    double sampling_rate;
    std::vector<double>* time = 0;
    std::vector<int>* id = 0;
    t->SetBranchAddress("sampling_rate", &sampling_rate);
    t->SetBranchAddress("count_x", &time);
    t->SetBranchAddress("id", &id);

    ofstream output(outfile);

    int nentries = t->GetEntries();
    int end = start + n > nentries ? nentries : start + n;
    if (n < 0) end = nentries;
    for (int i = start; i < end; i++) {
        if (i % 1000 == 0) cout << "Processing event " << i << " ..." << endl;
        t->GetEntry(i);

        std::vector<int> tvec(n_size, 0);
        int npri = 0;
        for (int j = 0; j < time->size(); j++) {
            int idx = (*time)[j]/sampling_rate;
            if (idx ==0 || idx >= n_size) continue;
            tvec[idx] = 1;
            if ((*id)[j] == 0) npri++;
        }

        output << setw(4) << npri;
        for (int j = 0; j < n_size; j++) {
            output << setw(2) << tvec[j];
        }
        output << endl;
    }

    output.close();
}
