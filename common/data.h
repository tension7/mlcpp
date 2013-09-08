#pragma once
#include <vector>
#include <string>
#include <cstdio>
#include <sstream>

void read_data(const std::string& feature_file, const std::string& data_file,
    std::vector<std::string>& fname,
    std::vector<int>& y, std::vector<std::vector<double>>& x) {

  FILE* f;

  char buf[10240];

  f = fopen(feature_file.c_str(), "r");
  while (fgets(buf, sizeof(buf), f)) {
    fname.push_back(buf);
    fname.back().pop_back();
  }
  fclose(f);
  int fn = fname.size(), lc = 0;
  f = fopen(data_file.c_str(),"r");
  while (fgets(buf, sizeof(buf), f)) {
    ++lc;
  }
  fclose(f);
  LOG(INFO) << "# of feature = " << fn << " # of samples = " << lc;
  x.resize(lc);
  for (auto& xx : x) { xx.resize(fn); }
  y.resize(lc);
  f = fopen(data_file.c_str(),"r");
  lc = 0;

  std::stringstream ss;
  while (fscanf(f, "%d", &y[lc]) == 1) {
    CHECK(y[lc] == 0 or y[lc] == 1) << "incorrect label";
    for (int i = 0; i < fn; ++i) {
      fscanf(f, "%lf", &x[lc][i]);
    }
    ++lc;
  }
  fclose(f);
}

void read_data_transpose(const std::string& feature_file, const std::string& data_file,
    std::vector<std::string>& fname,
    std::vector<int>& y, std::vector<std::vector<double>>& x) {

  FILE* f;

  char buf[10240];

  f = fopen(feature_file.c_str(), "r");
  while (fgets(buf, sizeof(buf), f)) {
    fname.push_back(buf);
    fname.back().pop_back();
  }
  fclose(f);
  int fn = fname.size(), lc = 0;
  x.resize(fn);
  f = fopen(data_file.c_str(),"r");
  while (fgets(buf, sizeof(buf), f)) {
    ++lc;
  }
  fclose(f);
  for (auto& xx : x) { xx.resize(lc); }
  y.resize(lc);
  f = fopen(data_file.c_str(),"r");
  lc = 0;
  while (fgets(buf, sizeof(buf), f)) {
    std::stringstream ss(buf);
    ss >> y[lc];
    CHECK(y[lc] == 0 or y[lc] == 1) << "incorrect label";
    for (int i = 0; i < fn; ++i) {
      ss >> x[i][lc];
    }
    ++lc;
  }
  fclose(f);
}


