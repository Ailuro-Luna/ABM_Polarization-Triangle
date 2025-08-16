from polarization_triangle.experiments.zealot_morality_analysis import run_no_zealot_morality_data


def main():
    # 以极小参数进行快速可运行性测试
    run_no_zealot_morality_data(
        output_dir="results/_smoke_zealot_morality",
        num_runs=1,
        max_morality=0,
        num_processes=1,
    )


if __name__ == "__main__":
    main()