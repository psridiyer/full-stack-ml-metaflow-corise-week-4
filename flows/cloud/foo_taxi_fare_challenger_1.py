
# Let's use a modified version of the event_triggered_linear_regression.py file from week 3 as the champion.
# Note that I encountered problems with reading the parquet url, and potentially saw some issues with caching - I had to explicitly state the parquet url and make sure
# to write the contents of the cell to a new file, using a different flow class name to make it work. Did not get to the root cause of the problem due to time limitations.

from metaflow import FlowSpec, step, card, conda_base, current, Parameter, Flow, trigger, project
from metaflow.cards import Markdown, Table, Image, Artifact


URL = 'https://outerbounds-datasets.s3.us-west-2.amazonaws.com/taxi/latest.parquet'
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

@project(name="foo_taxi_fare")
@trigger(events=['s3'])

@conda_base(libraries={'pandas': '1.4.2', 'pyarrow': '11.0.0', 'numpy': '1.21.2', 'scikit-learn': '1.1.2', 'xgboost': '1.7.4'})
class TaxiFarePrediction_Foo(FlowSpec):

    data_url = Parameter("data_url", default='https://outerbounds-datasets.s3.us-west-2.amazonaws.com/taxi/latest.parquet')

    def transform_features(self, df):

        # TODO: 
            # Try to complete tasks 2 and 3 with this function doing nothing like it currently is.
            # Understand what is happening.
            # Revisit task 1 and think about what might go in this function.
        obviously_bad_data_filters = [

            df.fare_amount > 0,         # fare_amount in US Dollars
            df.trip_distance <= 100,    # trip_distance in miles
            df.trip_distance > 0,
            df.passenger_count > 0,
            df.extra > 0,
            df.mta_tax > 0,
            df.tip_amount >= 0,
            df.tolls_amount >= 0,

            # TODO: add some logic to filter out what you decide is bad data!
            # TIP: Don't spend too much time on this step for this project though, it practice it is a never-ending process.

        ]

        for f in obviously_bad_data_filters:
            df = df[f]
        return df

    @step
    def start(self):

        import pandas as pd
        import io
        import requests
        from sklearn.model_selection import train_test_split

        self.df = self.transform_features(pd.read_parquet('https://outerbounds-datasets.s3.us-west-2.amazonaws.com/taxi/latest.parquet'))

        # NOTE: we are split into training and validation set in the validation step which uses cross_val_score.
        # This is a simple/naive way to do this, and is meant to keep this example simple, to focus learning on deploying Metaflow flows.
        # In practice, you want split time series data in more sophisticated ways and run backtests. 
        self.X = self.df["trip_distance"].values.reshape(-1, 1)
        self.y = self.df["total_amount"].values
        self.next(self.xgboost_model)

    @step
    def xgboost_model(self):
        from xgboost import XGBRegressor
        self.model = XGBRegressor()
        self.model_type = "xgboos"
        self.next(self.validate)

    def gather_sibling_flow_run_results(self):

        # storage to populate and feed to a Table in a Metaflow card
        rows = []

        # loop through runs of this flow 
        for run in Flow(self.__class__.__name__):
            if run.id != current.run_id:
                if run.successful:
                    icon = "✅" 
                    msg = "OK"
                    score = str(run.data.scores.mean())
                else:
                    icon = "❌"
                    msg = "Error"
                    score = "NA"
                    for step in run:
                        for task in step:
                            if not task.successful:
                                msg = task.stderr
                row = [Markdown(icon), Artifact(run.id), Artifact(run.created_at.strftime(DATETIME_FORMAT)), Artifact(score), Markdown(msg)]
                rows.append(row)
            else:
                rows.append([Markdown("✅"), Artifact(run.id), Artifact(run.created_at.strftime(DATETIME_FORMAT)), Artifact(str(self.scores.mean())), Markdown("This run...")])
        return rows
                
    
    @card(type="corise")
    @step
    def validate(self):
        from sklearn.model_selection import cross_val_score
        self.scores = cross_val_score(self.model, self.X, self.y, cv=5)
        current.card.append(Markdown("# Taxi Fare Prediction Results"))
        current.card.append(Table(self.gather_sibling_flow_run_results(), headers=["Pass/fail", "Run ID", "Created At", "R^2 score", "Stderr"]))
        self.next(self.end)

    @step
    def end(self):
        print("Success!")


if __name__ == "__main__":
    TaxiFarePrediction_Foo()
