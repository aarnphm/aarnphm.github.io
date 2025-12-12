#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

bool repeating(const char *str, int len, int pattern_len) {
    if (len % pattern_len != 0) {
        return false;
    }

    for (int i = 0; i < len; i++) {
        if (str[i] != str[i % pattern_len]) {
            return false;
        }
    }

    return true;
}

bool p1(long long num) {
    char str[32];
    sprintf(str, "%lld", num);
    int len = strlen(str);

    if (len % 2 != 0) {
        return false;
    }

    int half = len / 2;
    return strncmp(str, str + half, half) == 0;
}

bool p2(long long num) {
    char str[32];
    sprintf(str, "%lld", num);
    int len = strlen(str);

    for (int pattern_len = 1; pattern_len <= len / 2; pattern_len++) {
        if (len % pattern_len == 0) {
            int repetitions = len / pattern_len;
            if (repetitions >= 2 && repeating(str, len, pattern_len)) {
                return true;
            }
        }
    }

    return false;
}

int main() {
    FILE *fp = fopen("./d2.txt", "r");
    if (!fp) {
        perror("failed to open file");
        return 1;
    }

    char *line = NULL;
    size_t len = 0;
    ssize_t read;

    unsigned long long total_part1 = 0;
    unsigned long long total_part2 = 0;

    if ((read = getline(&line, &len, fp)) != -1) {
        char *token = strtok(line, ",");
        while (token != NULL) {
            long long start, end;
            if (sscanf(token, "%lld-%lld", &start, &end) == 2) {
                for (long long num = start; num <= end; num++) {
                    if (p1(num)) {
                        total_part1 += num;
                    }
                    if (p2(num)) {
                        total_part2 += num;
                    }
                }
            }
            token = strtok(NULL, ",");
        }
    }

    printf("part 1: %llu\n", total_part1);
    printf("part 2: %llu\n", total_part2);

    free(line);
    fclose(fp);
    return 0;
}
