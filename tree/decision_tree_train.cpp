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

DEFINE_string(data_file, "alldata.train", "");
DEFINE_int32(subsample100, 100, "");
DEFINE_int32(subsample_seed, 0, "");
DEFINE_string(model_file, "dt_model.alldata.train", "");
DEFINE_string(feature_file, "features", "");
DEFINE_int32(max_leaves, 10, "");
DEFINE_double(hash_weight, 1.4, "");

using namespace std;

inline double get_entropy(int p, int n) {
  double v = 1.0*p/(p+n);
  double e = 0;
  if (p > 0) { e += p*log(v); }
  if (n > 0) { e += n*log(1-v); }
  return -e;
}

vector<int> y;
vector<string> fname;
vector<vector<double>> x;
struct cache { double x; int id; int y; };
vector<cache> all;
vector<vector<cache>> idsbyf;
int fn;


struct node {
  double r;
  int fid;
  double split;
  int idstart, idend;
  int p, n, idx;
  shared_ptr<node> left, right;

  node(int _p, int _n) : r(0), p(_p), n(_n), idx(-1) {
  }

  node(int _idstart, int _idend, int _p, int _n) : idstart(_idstart), idend(_idend), p(_p), n(_n), idx(-1) {
    CHECK(p != 0 && n != 0) << "node initialized with single label";
    r = -1;
    if ((idend-idstart+1) * log(idend-idstart+1) > all.size() * FLAGS_hash_weight) {
      clock_t t0 = clock();
      unordered_set<int> iset;
      for (int i = idstart; i != idend; ++i) { iset.insert(all[i].id); }
      for (int i = 0; i < fn; ++i) {
        get_max_entropy_reduce2(iset, i);
      }
      VLOG(3) << "REDUCE1 " << idend-idstart + 1 << " " << clock() - t0;
    } else {
      clock_t t0 = clock();
      for (int i = 0; i < fn; ++i) {
        get_max_entropy_reduce(i);
      }
      VLOG(3) << "REDUCE2 " << idend-idstart + 1 << " " << clock() - t0;
    }
  }

  void get_max_entropy_reduce(int fid) {
    for (int i = idstart; i != idend; ++i) { all[i].x = x[fid][all[i].id]; }
    sort(all.begin()+idstart, all.begin()+idend,
         [&](const cache& a, const cache& b) { return a.x < b.x; });
    int pos = 0, neg = 0;
    double mine = get_entropy(p, n);
    double prevf = all[idstart].x - 1;
    double split = all[idstart].x - 0.5;
    for (int k = idstart; k != idend; ++k) {
      if (fabs(all[k].x-prevf) > 1e-10) {
        double e = get_entropy(pos, neg) + get_entropy(p-pos, n-neg);
        if (e < mine) {
          mine = e;
          split = (all[k].x + prevf) / 2;
        }
      }
      // split happens before i so update pos and neg afterwards
      if (all[k].y == 1) { pos++; } else { neg++; }
      prevf = all[k].x;
    }
    double r = get_entropy(p, n) - mine;
    if (r > this->r) {
      this->r = r;
      this->fid = fid;
      this->split = split;
    }
  }

  template<typename SetT>
  void get_max_entropy_reduce2(const SetT& ids, int fid) {
    int pos = 0, neg = 0;
    double mine = get_entropy(p, n);
    double prevf = idsbyf[fid][0].x - 1;
    double split = idsbyf[fid][0].x - 0.5;
    for (auto& ic : idsbyf[fid]) {
      if (ids.count(ic.id) == 0) { continue; } // not in ids
      if (fabs(ic.x-prevf) > 1e-10) {
        double e = get_entropy(pos, neg) + get_entropy(p-pos, n-neg);
        if (e < mine) {
          mine = e;
          split = (ic.x + prevf) / 2;
        }
      }
      // split happens before i so update pos and neg afterwards
      if (ic.y == 1) { pos++; } else { neg++; }
      prevf = ic.x;
    }
    double r = get_entropy(p, n) - mine;
    if (r > this->r) {
      this->r = r;
      this->fid = fid;
      this->split = split;
    }
  }
};

void write_model(const vector<shared_ptr<node>>& nodes) {
  /// mark the indices
  for (int i = 0; i < nodes.size(); ++i) {
    nodes[i]->idx = i;
  }
  FILE *f = fopen(FLAGS_model_file.c_str(), "w");
  // print the tree
  unordered_map<string, double> fi, fir;
  for (auto& i : nodes) {
    if (i->left) {
      fprintf(f, "%d 0 %d %lf %d %d\n", i->idx, i->fid, i->split, i->left->idx, i->right->idx);
      fi[fname[i->fid]] += i->r;
      fir[fname[i->fid].substr(0, fname[i->fid].find(':'))] += i->r;
    } else {
      fprintf(f, "%d 1 %lf\n", i->idx, 1.0*i->p / (i->p + i->n));
    }
    VLOG(2) << "feature: " << i->fid << " reduction = " << i->r << " " << i->split << " " << i->p << " " << i->n;
  }
  fclose(f);
  vector<pair<string, double>> fiv(fi.begin(), fi.end()), firv(fir.begin(), fir.end());
  auto ficomp = [](const pair<string,double>&a, const pair<string,double>&b) { return a.second > b.second; };
  sort(fiv.begin(), fiv.end(), ficomp);
  sort(firv.begin(), firv.end(), ficomp);
  LOG(INFO) << "feature importance:";
  for (auto& f : fiv) {
    LOG(INFO) << "feature " << f.first << " total reduction = " << f.second;
  }
  LOG(INFO) << "feature importance roll-up:";
  for (auto& f : firv) {
    LOG(INFO) << "feature " << f.first << " total reduction = " << f.second;
  }
}


void train(vector<shared_ptr<node>>& nodes) {
  // presort each feature dimension to speed up
  if (FLAGS_subsample_seed) {
    srand(FLAGS_subsample_seed);
  }
  all.resize(x[0].size());
  for (int i = 0; i < all.size(); ++i) { all[i].id = i; }
  if (FLAGS_subsample100 < 100) {
    random_shuffle(all.begin(), all.end());
    all.resize(all.size() * FLAGS_subsample100/100);
  }
  int totp = 0, totn = 0, tot = all.size();
  for (auto& ic : all) {
    ic.y = y[ic.id];
    if (ic.y == 1) { ++totp; } else { ++totn; }
  }
  CHECK(totp + totn == tot);
  LOG(INFO) << "pos = " << totp << " neg = " << totn << " total = " << tot << " feature = " << fn;

  idsbyf.resize(fn);
  for (int i = 0; i < fn; ++i) {
    idsbyf[i].resize(all.size());
    for (int j = 0; j < all.size(); ++j) {
      idsbyf[i][j].x = x[i][all[j].id];
      idsbyf[i][j].id = all[j].id;
      idsbyf[i][j].y = all[j].y;
    }
    sort(idsbyf[i].begin(), idsbyf[i].end(), [&](const cache& a, const cache& b) { return a.x < b.x; });
  }

  struct comp {
    bool operator()(const shared_ptr<node>& a, const shared_ptr<node> &b) const { return a->r < b->r; }
  };
  typedef priority_queue<shared_ptr<node>, vector<shared_ptr<node>>, comp> queue_type;
  //typedef queue<shared_ptr<node>> queue_type;
  auto get_front = [](queue_type& q) -> const queue_type::value_type& { return q.top(); };
  //auto get_front = queue_type::value_type& [](queue_type& q) { return q.front(); }
  queue_type cand;

  cand.push(make_shared<node>(0, tot, totp, totn));
  nodes.push_back(get_front(cand));

  int s = 0, n = 0;
  while (n < FLAGS_max_leaves && !cand.empty()) {
    auto cur = get_front(cand);
    cand.pop();
    ++n;
    // now split cur->ids based on cur->split
    auto fid = cur->fid;
    auto split = cur->split;
    LOG(INFO) << "PICK: reduction = " << cur->r << " feature #" << fid << " split = " << split;

    for (int i = cur->idstart; i != cur->idend; ++i) {
      all[i].x = x[fid][all[i].id];
    }
    int lpos = 0, lneg = 0, l = cur->idstart, r = cur->idend - 1;
    while (l <= r) {
      while (all[l].x < split && l <= r) ++l;
      while (all[r].x >= split && l <= r) --r;
      if (l < r) {
        swap(all[l++], all[r--]);
      }
    }
    for (int k = cur->idstart; k != l; ++k) {
      if (all[k].y == 1) { lpos++; } else { lneg++; }
    }
    if (lpos == 0 || lneg == 0) {
      //perfect split is done, no subnodes are needed
      cur->left = make_shared<node>(lpos, lneg);
      VLOG(2) << "STOP LEFT: pos:neg = " << lpos <<  ":" << lneg;
    } else {
      cur->left = make_shared<node>(cur->idstart, l, lpos, lneg);
      VLOG(2) << "INSERT LEFT: reduction = " << cur->left->r <<  " feature #" << cur->left->fid << " split = " << cur->left->split
                << " pos:neg = " << lpos << ":" << lneg;
    }
    if (cur->left->r > 0) { cand.push(cur->left); } else { s += cur->left->p + cur->left->n; }
    nodes.push_back(cur->left);
    if (cur->p - lpos == 0 || cur->n - lneg == 0) {
      //perfect split is done, no subnodes are needed
      cur->right = make_shared<node>(cur->p - lpos, cur->n - lneg);
      VLOG(2) << "STOP RIGHT: pos:neg = " << cur->p - lpos << ":" << cur->n - lneg;
    } else {
      cur->right = make_shared<node>(l, cur->idend, cur->p - lpos, cur->n - lneg);
      VLOG(2) << "INSERT RIGHT: reduction = " << cur->right->r <<  " feature #" << cur->right->fid << " split = " << cur->right->split
                << " pos:neg = " << cur->p-lpos << ":" << cur->n-lneg;
    }
    if (cur->right->r > 0) { cand.push(cur->right); } else { s += cur->right->p + cur->right->n; }
    nodes.push_back(cur->right);
  }
  LOG(INFO) << "unexplored nodes = " << cand.size() << " with max entropy reduction = " << (cand.empty() ? 0 : get_front(cand)->r);
  while(!cand.empty()) {
    s += get_front(cand)->p + get_front(cand)->n;
    cand.pop();
  }
  LOG(INFO) << "total samples = " << s;
  CHECK(s == tot);
}

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  read_data_transpose(FLAGS_feature_file, FLAGS_data_file, fname, y, x);
  fn = fname.size();
  vector<shared_ptr<node>> nodes;
  train(nodes);
  write_model(nodes);
  return 0;
}
