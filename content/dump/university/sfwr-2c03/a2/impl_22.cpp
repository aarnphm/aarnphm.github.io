#include <array>    /* Fixed-size arrays. */
#include <chrono>   /* Measuring time. */
#include <vector>   /* A vector is a dynamic array. */
#include <string>   /* Text values. */
#include <iostream> /* For printing output. */

/* Structure of a single course (a row in the courses relation). */
struct course
{
    std::array<char, 8> prog;
    std::array<char, 5> code;
    std::string name;
};

/* Structure of a single enrolled student (a row in the enrolled relation). */
struct enroll
{
    std::array<char, 8> prog;
    std::array<char, 5> code;
    unsigned sid;
};

/* Structure used as output by the join algorithms. */
struct output_pair
{
    unsigned sid;
    const std::string& name; /* Reference to the course name in the courses
                                table, no need to make expensive copies of long
                                strings. */
};


/*
 * The CEJoin algorithm.
 */
std::vector<output_pair> cejoin(const std::vector<course>& courses,
                                const std::vector<enroll>& enrolled)
{
    std::vector<output_pair> output;
    for (auto& [p_c, c_c, n_c] : courses) {
        for (auto& [p_e, c_e, s_e] : enrolled) {
            if (p_c == p_e and c_c == c_e) {
                /* Pushes a value to the back of the dynamic array. */
                output.emplace_back(s_e, n_c);
            }
        }
    }
    return output;
}

/*
 * The ECJoin algorithm.
 */
std::vector<output_pair> ecjoin(const std::vector<course>& courses,
                                const std::vector<enroll>& enrolled)
{
    std::vector<output_pair> output;
    for (auto& [p_e, c_e, s_e] : enrolled) {
        for (auto& [p_c, c_c, n_c] : courses) {
            if (p_c == p_e and c_c == c_e) {
                /* Pushes a value to the back of the dynamic array. */
                output.emplace_back(s_e, n_c);
            }
        }
    }
    return output;
}
                
/*
 * Measure the performance of the algorithm @{algorithm} when operating on the
 * inputs @{courses} and @{enrolled}. Print to output the time it takes to
 * run the algorithm and the output size of the algorithm (this output size can
 * be ignored).
 */
void measure_with(const std::vector<course>& courses,
                  const std::vector<enroll>& enrolled,
                  auto algorithm)
{
    using namespace std::chrono;

    auto start = system_clock::now();
    auto result = algorithm(courses, enrolled);
    auto end = system_clock::now();

    std::cout << "\t" << duration_cast<microseconds>(end - start).count()
              << "\t" << result.size();
    /* We print the result size to make sure we use the result of the call to
     * @{join_algorithm}, we do not want this call to be optimized away! */
}


/*
 * Return an enrolled relation enrolling @{num_students} to a course.
 */
std::vector<enroll> make_enrollment(const std::vector<course>& courses,
                                    std::size_t num_students)
{
    std::vector<enroll> enrolled;

    /* Index in the courses relation: determines the course we will enroll the
     * next student in. */
    std::size_t i = 0;

    while (enrolled.size() < num_students) {
        /* Pushes a value to the back of the dynamic array. */
        enrolled.emplace_back(courses[i].prog, courses[i].code, enrolled.size());

        ++i;

        if (i == courses.size()) {
            /* We only have a few courses, so restart at the beginning if we
             * reach the end. */
            i = 0;
        }
    }
    
    return enrolled;
}

/*
 * Measure the performance of cejoin and ecjoin with @{size} rows in the
 * enrolled relation (and six courses).
 */
void measure(std::size_t size)
{
    std::vector<course> courses = {
        {"COMPSCI", "2LC3", "Logical Reasoning for Computer Science"},
        {"COMPSCI", "2DB3", "Databases"},
        {"SFWRENG", "2C03", "Data Structures And Algorithms"},
        {"SFWRENG", "2DB3", "Databases"},
        {"SFWRENG", "2DM3", "Discrete Mathematics with Applications I"},
        {"SFWRENG", "4AD3", "Advanced Databases"}
    };
    std::vector<enroll> enrolled = make_enrollment(courses, size);

    std::cout << size;
    
    /* We run each algorithm twice in a row to increase the chance that one of
     * these measurements is free of unintended noise. We kept the minimum
     * measurement in the graph shown in the assignment. */    
    measure_with(courses, enrolled, cejoin);
    measure_with(courses, enrolled, cejoin);    
    measure_with(courses, enrolled, ecjoin);
    measure_with(courses, enrolled, ecjoin);
    std::cout << std::endl;
}


/*
 * Entry point of the program---here, execution starts.
 */
int main(int argc, char* argv[])
{
    std::cout << "Size\tCEJoin Runtime\t(ignore)\tCEJoin Runtime\t(ignore)"
                     "\tECJoin Runtime\t(ignore)\tECJoin Runtime\t(ignore)" << std::endl;
    
    for (std::size_t i = 0; i < 25'000'000; i += 500'000) {
        measure(i);
    }
}