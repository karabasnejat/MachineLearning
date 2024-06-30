# Decision Tree Classifier Project

This project involves implementing a Decision Tree Classifier from scratch in Python and evaluating its performance on a given dataset. The classifier is tested on both training and test datasets, and the results are saved in a text file. Additionally, the decision tree is visualized and saved as a PNG file.

## Prerequisites

Before running the code, make sure you have the following libraries installed:
- pandas
- numpy
- scikit-learn
- graphviz

You can install these libraries using pip:
```bash
pip install pandas numpy scikit-learn graphviz
```

## Project Structure

- `trainSet.csv`: Training dataset.
- `testSet.csv`: Test dataset.
- `CART.ipynb`: Jupyter Notebook containing the entire code.
- `performans_olcumleri.txt`: Text file where the performance metrics are saved.
- `decision_tree.gv` and `decision_tree.png`: Files for the visual representation of the decision tree.

## Steps

### 1. Load and Preprocess Data

The datasets are loaded using pandas. Categorical features are encoded into numerical values using `LabelEncoder`.

### 2. Decision Tree Classifier Implementation

A custom Decision Tree Classifier is implemented with methods to fit the model, predict labels, and find the best splits based on Gini impurity.

### 3. Training the Model

The classifier is trained on the training dataset with specified hyperparameters:
- `max_depth=3`
- `min_samples_split=10`
- `min_samples_leaf=5`

### 4. Evaluating the Model

The model's performance is evaluated on both training and test datasets. Metrics such as accuracy, true positive rate, and true negative rate are calculated and saved in `performans_olcumleri.txt`.

### 5. Visualizing the Decision Tree

The decision tree is visualized using the `graphviz` library and saved as a PNG file.

## How to Run

1. Place the `trainSet.csv` and `testSet.csv` files in the same directory as the Jupyter Notebook.
2. Open the `CART.ipynb` notebook and run all cells.
3. The performance metrics will be printed on the console and saved in `performans_olcumleri.txt`.
4. The decision tree will be visualized and saved as `decision_tree.png`.

## Output

- **Performance Metrics:** Accuracy, True Positive Rate, and True Negative Rate for both training and test datasets.
- **Decision Tree Visualization:** The structure of the trained decision tree.

## Example Results

The results of running the notebook on the sample data are saved in `performans_olcumleri.txt`. Here is an example of what you can expect:

```
Eğitim Sonuçları:
Accuracy: 0.85
True Positive Rate: 0.88
True Negative Rate: 0.83
True Positive Adedi: 44
True Negative Adedi: 39

Test Sonuçları:
Accuracy: 0.82
True Positive Rate: 0.86
True Negative Rate: 0.79
True Positive Adedi: 43
True Negative Adedi: 37
```

## Notes

- Ensure the paths to the CSV files are correct.
- The feature names should match those in your dataset for the decision tree visualization to be accurate.
- Adjust the hyperparameters as needed to improve the model's performance.

## Author

**Oğuzhan Nejat Karabaş** 

## License

This project is licensed under the MIT License.

---

Feel free to customize and enhance the README as needed for your specific use case and audience.
