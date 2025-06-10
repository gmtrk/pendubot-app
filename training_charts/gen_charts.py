import pandas as pd
import matplotlib.pyplot as plt
import os

csv_files = [
    'train_entropy_loss.csv',
    'train_value_loss.csv',
    'rollout_ep_len_mean.csv',
    'rollout_ep_rew_mean.csv'
]

plot_titles_polish = {
    'train_entropy_loss.csv': 'Strata Entropii Treningowej',
    'train_value_loss.csv': 'Strata Wartości Treningowej',
    'rollout_ep_len_mean.csv': 'Średnia Długość Epizodu (Rollout)',
    'rollout_ep_rew_mean.csv': 'Średnia Nagroda Epizodu (Rollout)'
}

output_dir = '../old_runs/charts'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Generating charts to : {os.path.abspath(output_dir)}")

TITLE_FONT_SIZE = 24
LABEL_FONT_SIZE = 18
TICK_FONT_SIZE = 16
LEGEND_FONT_SIZE = 16

for file_name in csv_files:
    try:
        df = pd.read_csv(file_name)

        if 'Step' not in df.columns or 'Value' not in df.columns:
            print(f"Warning: File '{file_name}' does not contain 'Step' or 'Value' columns. Skipping.")
            continue

        steps = df['Step']
        values = df['Value']

        plot_title = plot_titles_polish.get(file_name, file_name.replace('_', ' ').replace('.csv', '').title())
        output_file_name = os.path.join(output_dir, file_name.replace('.csv', '.png'))

        plt.figure(figsize=(10, 6))
        plt.plot(steps, values)

        plt.title(plot_title, fontsize=TITLE_FONT_SIZE)

        plt.xlabel('Kroki', fontsize=LABEL_FONT_SIZE)
        plt.ylabel('Wartość', fontsize=LABEL_FONT_SIZE)

        plt.tick_params(axis='x', labelsize=TICK_FONT_SIZE)
        plt.tick_params(axis='y', labelsize=TICK_FONT_SIZE)

        plt.grid(True)
        plt.tight_layout()

        plt.savefig(output_file_name)
        print(f"Generated and saved: {output_file_name}")

        plt.close()

    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found. Make sure it is in the same directory as the script.")
    except Exception as e:
        print(f"An error occurred while processing the file '{file_name}': {e}")

print("\nChart generation process finished.")