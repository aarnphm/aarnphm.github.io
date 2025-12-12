class Solution {
public:
  int maxCoins(vector<int> &nums) {
    int n = nums.size();
    vector<int> a(n + 2);

    a[0] = a[n + 1] = 1;
    for (int i = 0; i < n; i++)
      a[i + 1] = nums[i];

    vector<vector<int>> dp(n + 2, vector<int>(n + 2, 0));

    for (int len = 1; len <= n; len++) {
      for (int i = 1; i + len <= n + 1; i++) {
        int j = i + len - 1;
        for (int k = i; k <= j; k++) {
          dp[i][j] = max(dp[i][j], dp[i][k - 1] + a[i - 1] * a[k] * a[j + 1] +
                                       dp[k + 1][j]);
        }
      }
    }
    return dp[1][n];
  }

  // NOTE: I was thinking initially to improve upon previous implementation.
  // What we can really do here?
  // - memory locality: probably a flat array works, given that the constraint
  //                    is 300, hence dp[305][305] should work
  //                    or a single vector `vector<int> dp((n+2)*(n+2))` should
  //                    also work.
  // - reduce constant: we can probably precompute some fo these as well,
  //                    i.e lr = a[i=1] * a[j+1] once
  //                    (so we save like one cycle lol)
  // - optimization: monotone, Knuth et al. require
  //                           cost to have specific structures.
  //                           Our cost term is `a[i-1] * a[k] * a[j+1]`, which
  //                           depends on k.
  //                           so O(n^2)  interval-DP-optimization
  //                           won't work here.
  int maxCoinsAsymptotic(vector<int> &nums) {
    vector<int> a;
    a.reserve(nums.size() + 2);
    a.push_back(1);
    for (int x : nums)
      if (x != 0)
        a.push_back(x);
    a.push_back(1);

    int n = (int)a.size() - 2;
    if (n <= 0)
      return 0;

    static int dp[305][305];
    for (int i = 0; i <= n + 1; ++i)
      for (int j = 0; j <= n + 1; ++j)
        dp[i][j] = 0;

    for (int len = 1; len <= n; ++len) {
      for (int i = 1; i + len - 1 <= n; ++i) {
        int j = i + len - 1;
        int lr = a[i - 1] * a[j + 1];
        int best = 0;
        for (int k = i; k <= j; ++k) {
          best = max(best, dp[i][k - 1] + lr * a[k] + dp[k + 1][j]);
        }
        dp[i][j] = best;
      }
    }
    return dp[1][n];
  }
};
