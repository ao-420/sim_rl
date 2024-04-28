from foundations.core_functions import start_train, start_tuning
import argparse


def main(args):
    if args.function == "tune":
        project_name = "datasparq"
        wandb_api_key = (
            "02bb2e4979e9df3d890f94a917a95344aae652b9"  # replace with yours here
        )
        num_runs = 10
        plot_best_param = False
        start_tuning(
            project_name,
            num_runs,
            args.param_file,
            args.config_file,
            args.param_file,
            wandb_api_key,
            plot_best_param,
            tuner=args.tuner,
        )
    elif args.function == "train":
        start_train(
            args.config_file,
            args.param_file,
            data_filename=args.data_file,
            image_filename=args.image_file,
            save_file=args.save_file,
            plot_curves=args.plot_curves,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training or tuning")
    parser.add_argument(
        "--function",
        choices=["train", "tune"],
        required=True,
        help="Function to run: train or tune",
    )
    parser.add_argument(
        "--config_file", required=True, help="File path for configuration parameters"
    )
    parser.add_argument(
        "--param_file",
        required=True,
        help="File path for evaluation or tuning parameters",
    )
    parser.add_argument(
        "--data_file", default="output_csv", help="Filename for saving output data"
    )
    parser.add_argument(
        "--image_file", default="output_plots", help="Filename for saving output images"
    )
    parser.add_argument(
        "--save_file", type=bool, default=True, help="Flag to save output files"
    )
    parser.add_argument(
        "--plot_curves", type=bool, default=True, help="Flag to plot generated curves"
    )
    parser.add_argument(
        "--tuner", required=False, help="Flag to choose which tuner to use"
    )

    args = parser.parse_args()
    main(args)
