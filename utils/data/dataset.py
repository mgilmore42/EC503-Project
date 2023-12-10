import polars as pl

class Dataset:

    def __init__(self, fname: str):
        """
        Initialize a Dataset object.

        This method initializes a Dataset object by reading the dataset from the specified file
        name. It also normalizes the numerical columns and converts the categorical columns to
        numerical representation.

        Args:
            fname (str): The file name of the dataset.

        Returns:
            None
        """

        self.df = pl.read_csv(fname, null_values=['NA'])

        self._normalize()
        self._categorical2numerical()


    def _categorical2numerical(self):
        """
        Convert categorical columns in the DataFrame to numerical representation.

        This method converts each categorical column in the DataFrame to a numerical representation
        by assigning a unique index to each category. It also stores the mapping between the index
        and the original category in the `categorical` dictionary.

        Returns:
            None
        """
        self.categorical = {}
        
        for col in self.df.columns:
            if self.df[col].dtype == pl.Utf8:
                categories = self.df[col].unique().to_list()

                self.df = self.df.with_columns(
                    self.df[col].apply(lambda x: categories.index(x))
                )

                self.categorical[col] = {i: category for i, category in enumerate(categories)}
    
    def _make_categorical(self, col: str):
        """
        Convert a numerical column in the DataFrame to categorical representation.

        This method converts a numerical column in the DataFrame to a categorical representation
        by assigning the original category to each index. It uses the `categorical` dictionary to
        perform the conversion.

        Args:
            col (str): The column to convert to categorical representation.

        Returns:
            None
        """
        
        categories = self.df[col].unique().to_list()
        categories.sort()

        self.df = self.df.with_columns(
            self.df[col].apply(lambda x: categories.index(x))
        )

        self.categorical[col] = {i: category for i, category in enumerate(categories)}
    
    def _normalize(self):
        """
        Normalize the numerical columns in the DataFrame.
        
        This method normalizes each numerical column in the DataFrame by subtracting the mean and
        dividing by the standard deviation.
        
        Returns:
            None
        """

        for col in self.df.columns:
            if self.df[col].dtype == pl.Float64:

                self.df = self.df.with_columns((self.df[col] - self.df[col].mean()) / self.df[col].std())
    
    def _make_normalization(self, col: str):
        """
        Normalize a numerical column in the DataFrame.
        
        This method normalizes a numerical column in the DataFrame by subtracting the mean and
        dividing by the standard deviation.
        
        Args:
            col (str): The column to normalize.
        
        Returns:
            None
        """

        data = self.df[col].cast(pl.Float64)

        self.df = self.df.with_columns((data - data.mean()) / data.std())
    
    def get(self, gt: list[str], features: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Get the dataset as a tuple of features and ground-truth.

        This method returns the dataset as a tuple of features and ground-truth. The features are
        specified by the `features` argument, and the ground-truth is specified by the `gt` argument.

        Args:
            gt (list[str]): The ground-truth columns to return.
            features (list[str]): The feature columns to return.

        Returns:
            tuple[pl.DataFrame, pl.DataFrame]: A tuple of features and ground-truth.
        """
        return self.df[features], self.df[gt]
    
    def _remove(self, col: str, *args: str):
        """
        Remove a column from the DataFrame.

        This method removes a column from the DataFrame.

        Args:
            col (str): The column to remove.

        Returns:
            None
        """

        self.df = self.df.drop(col, *args)
    
    def save(self, fname: str):
        """
        Save the DataFrame to a CSV file.

        This method saves the DataFrame to a CSV file.

        Args:
            fname (str): The file name to save the DataFrame to.

        Returns:
            None
        """

        self.df.to_csv(fname)

class CampusDataset(Dataset):

    def __init__(self):
        """
        Initialize a CampusDataset object.

        This method initializes a CampusDataset object by reading the dataset from the specified
        file name. It also normalizes the numerical columns and converts the categorical columns to
        numerical representation.

        Returns:
            None
        """

        super().__init__('data/campus/Placement_Data_Full_Class.csv')
        self._remove('sl_no')
        self._make_normalization('salary')

class HeartDataset(Dataset):

    def __init__(self):
        """
        Initialize a HeartDataset object.

        This method initializes a HeartDataset object by reading the dataset from the specified
        file name. It also normalizes the numerical columns and converts the categorical columns to
        numerical representation.

        Returns:
            None
        """

        super().__init__('data/heart/heart.csv')

        self._make_normalization('age')
        self._make_normalization('trtbps')
        self._make_normalization('chol')
        self._make_normalization('thalachh')

class HousingDataset(Dataset):

    def __init__(self):
        """
        Initialize a HousingDataset object.

        This method initializes a HousingDataset object by reading the dataset from the specified
        file name. It also normalizes the numerical columns and converts the categorical columns to
        numerical representation.

        Returns:
            None
        """

        super().__init__('data/housing/housePrices.csv')

        self._make_categorical('Zip')
        self._make_normalization('Year')
        self._make_categorical('Beds')
        self._make_categorical('Baths')
        self._make_normalization('Living_Space_sq_ft')
        self._make_normalization('SalePrice')

        self._remove('Index')

class RainDataset(Dataset):

    def __init__(self):
        """
        Initialize a RainDataset object.

        This method initializes a RainDataset object by reading the dataset from the specified
        file name. It also normalizes the numerical columns and converts the categorical columns to
        numerical representation.

        Returns:
            None
        """

        super().__init__('data/rain/weatherAUS.csv')

        self._remove('Date', 'Evaporation', 'Sunshine', 'Cloud3pm', 'Cloud9am')
        self._make_normalization('WindGustDir')
        self._make_normalization('WindGustSpeed')
        self._make_normalization('WindDir9am')
        self._make_normalization('WindSpeed9am')
        self._make_normalization('WindDir3pm')
        self._make_normalization('WindSpeed3pm')
        self._make_normalization('Humidity9am')
        self._make_normalization('Humidity3pm')