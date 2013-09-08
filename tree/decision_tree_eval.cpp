#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <vector>
#include <sstream>
#include <memory>
#include "external/gflags/gflags.h"
#include "external/glog/logging.h"
#include "common/data.h"

DEFINE_string(data_file, "alldata.test", "");
DEFINE_string(feature_file, "features", "");
DEFINE_string(model_file, "dt_model.alldata.train", "");
DEFINE_int32(bagging, 0, "");
DEFINE_bool(transform, false, "");
DEFINE_int32(subsample100, 100, "");

using namespace std;

struct eval_node {
  int fid;
  double split;
  int left, right;
};

void read_model(const string& fname, vector<eval_node>& nodes) {
  FILE *f = fopen(fname.c_str(), "r");
  // print the tree
  char buf[10240];
  int idx = 0;
  while (fgets(buf, sizeof(buf), f)) {
    stringstream ss(buf);
    int tmp, type;
    ss >> tmp >> type;
    CHECK (tmp == idx);
    nodes.emplace_back();
    eval_node* cur = &nodes.back();
    if (type == 0) {
      ss >> cur->fid >> cur->split >> cur->left >> cur->right;
    } else {
      cur->fid = cur->left = cur->right = -1;
      ss >> cur->split;
    }
    ++idx;
  }
  fclose(f);
}

double eval(const vector<double> f, const vector<eval_node>& nodes, int* ind = nullptr) {
  const eval_node* cur;
  for (cur = &nodes[0]; cur->fid != -1; ) {
    if (f[cur->fid] < cur->split) {
      cur = &nodes[cur->left];
    } else {
      cur = &nodes[cur->right];
    }
  }
  if (ind) { *ind = (int)(cur - &nodes[0]); }
  return cur->split;//rand()%1000/1000.0 < cur->split ? 1 : 0; //cur->split > 0.5 ? 1 : 0;
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  vector<int> y, yy;
  vector<string> fname;
  vector<vector<double>> x;
  read_data(FLAGS_feature_file, FLAGS_data_file, fname, y, x);

  vector<int> all(x.size());
  for (int i = 0; i < all.size(); ++i) { all[i] = i; }
  if (FLAGS_subsample100 != 100) { random_shuffle(all.begin(), all.end()); }
  all.resize(all.size() * FLAGS_subsample100/100);

  vector<eval_node> nodes;

  if (FLAGS_transform && FLAGS_bagging) {
    vector<double> yy(x.size());
    vector<vector<pair<int,double>>> tr(all.size());
    for (int i = 0; i < FLAGS_bagging; ++i) {
      stringstream fn;
      fn << FLAGS_model_file << "_" << i+1;
      nodes.clear();
      read_model(fn.str(), nodes);
      int k = 0;
      for (auto i : all ) {
        int ind;
        double y = eval(x[i], nodes, &ind);
        tr[k++].push_back(make_pair(ind, y));
      }
    }
    for (auto& r : tr) {
      for (auto& f : r) {
        cout << f.first << ":" << f.second << " ";
      }
      cout << endl;
    }
    return 0;
  }

  if (!FLAGS_bagging) {
    read_model(FLAGS_model_file, nodes);
    for (auto i : all) {
      cout << y[i] << " " << eval(x[i], nodes) << endl;
    }
  } else {
    vector<double> yy(x.size());
    for (int i = 0; i < FLAGS_bagging; ++i) {
      stringstream fn;
      fn << FLAGS_model_file << "_" << i+1;
      nodes.clear();
      read_model(fn.str(), nodes);
      for (auto i : all) {
        yy[i] += eval(x[i], nodes);
      }
    }
    for (auto i : all) {
      cout << y[i] << " " << yy[i]/FLAGS_bagging << endl;
    }
  }
  return 0;
}
