#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <algorithm>
#include <climits>

using namespace std;

struct Shape {
  vector<pair<int,int>> cells;
  int minR, maxR, minC, maxC;

  void normalize() {
    minR = minC = INT_MAX;
    maxR = maxC = INT_MIN;
    for (auto& [r,c] : cells) {
      minR = min(minR, r); maxR = max(maxR, r);
      minC = min(minC, c); maxC = max(maxC, c);
    }
    for (auto& [r,c] : cells) { r -= minR; c -= minC; }
    sort(cells.begin(), cells.end());
    maxR -= minR; maxC -= minC; minR = minC = 0;
  }

  Shape rotate() const {
    Shape s;
    for (auto [r,c] : cells) s.cells.push_back({c, -r});
    s.normalize();
    return s;
  }

  Shape flip() const {
    Shape s;
    for (auto [r,c] : cells) s.cells.push_back({r, -c});
    s.normalize();
    return s;
  }
};

int W, H;
vector<vector<Shape>> all;
vector<vector<bool>> grid;

bool can(const Shape& s, int r, int c) {
  if (r + s.maxR >= H || c + s.maxC >= W)  return false;
  for (auto [dr, dc]: s.cells)
    if (grid[r+dr][c+dc]) return false;
  return true;
}

void place(const Shape& s, int r, int c, bool val) {
  for (auto [dr,dc] : s.cells) grid[r+dr][c+dc] = val;
}

bool backtrack(vector<int>& counts, int shape_idx) {
  while (shape_idx < 6 && counts[shape_idx] == 0) shape_idx++;
  if (shape_idx == 6) return true;

  counts[shape_idx]--;
  for (const Shape& s : all[shape_idx]) {
    for (int r = 0; r < H; r++) {
      for (int c = 0; c < W; c++) {
        if (can(s, r, c)) {
          place(s, r, c, true);
          if (backtrack(counts, shape_idx)) {
            place(s, r, c, false);
            counts[shape_idx]++;
            return true;
          }
          place(s, r, c, false);
        }
      }
    }
  }
  counts[shape_idx]++;
  return false;
}

int main() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);

  ifstream fin("d12.txt");
  string line;

  vector<Shape> shapes(6);
  for (int i = 0; i < 6; i++) {
    getline(fin, line);
    for (int r = 0; r < 3; r++) {
      getline(fin, line);
      for (int c = 0; c < 3; c++)
        if (line[c] == '#') shapes[i].cells.push_back({r, c});
    }
    shapes[i].normalize();
    getline(fin, line);
  }

  all.resize(6);
  for (int i = 0; i < 6; i++) {
    set<vector<pair<int,int>>> seen;
    Shape s = shapes[i];
    for (int f = 0; f < 2; f++) {
      for (int rt = 0; rt < 4; rt++) {
        if (seen.insert(s.cells).second) all[i].push_back(s);
        s = s.rotate();
      }
      s = s.flip();
    }
  }

  int answer = 0;
  while (getline(fin, line)) {
    if (line.empty()) continue;
    int x = line.find('x'), colon = line.find(':');
    W = stoi(line.substr(0, x));
    H = stoi(line.substr(x+1, colon-x-1));

    vector<int> counts(6);
    istringstream iss(line.substr(colon+1));
    int total = 0;
    for (int i = 0; i < 6; i++) {
      iss >> counts[i];
      total += counts[i] * (int)shapes[i].cells.size();
    }

    if (total > W * H) continue;

    grid.assign(H, vector<bool>(W, false));
    if (backtrack(counts, 0)) answer++;
  }

  cout << answer << "\n";
  return 0;
}
