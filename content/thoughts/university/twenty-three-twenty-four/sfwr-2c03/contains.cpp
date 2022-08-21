#include <chrono>
#include <cstddef>
#include <iostream>
#include <vector>

bool find(const std::vector<int> &list, int value) {
  bool found = false;
  std::size_t i = 0;
  while (i != list.size()) {
    if (list[i] == value) {
      found = true;
      ++i;
    } else {
      ++i;
    }
  }
  return found;
}

std::vector<int> generate_list(std::size_t size) {
  std::vector<int> table;
  while (table.size() != size) {
    table.emplace_back(table.size());
  }
  return table;
}

void measure_find(std::size_t size) {
  using namespace std::chrono;
  auto list = generate_list(size);
  auto value = size; /* not in list. */

  auto start = steady_clock::now();
  auto found = find(list, value);
  auto end = steady_clock::now();
  auto measurement = duration_cast<microseconds>(end - start);

  /* Nicely print output. */
  std::cout << size << '\t' << measurement.count() << '\t' << found << '\n';
}

int main(int argc, char *argv[]) {
  for (std::size_t size = 0; size <= 1024 * 1024 * 32; size += 1024 * 1024) {
    measure_find(size);
  }
}
