package com.example;

import com.google.common.base.Splitter;
import org.tribuo.*;
import org.tribuo.classification.sgd.objectives.LogMulticlass;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.evaluation.CrossValidation;
import org.tribuo.evaluation.TrainTestSplitter;
import org.tribuo.impl.ArrayExample;
import org.tribuo.math.optimisers.AdaGrad;
import org.tribuo.regression.RegressionFactory;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.evaluation.RegressionEvaluation;
import org.tribuo.regression.evaluation.RegressionEvaluator;
import org.tribuo.regression.liblinear.LibLinearRegressionTrainer;
import org.tribuo.regression.sgd.RegressionObjective;
import org.tribuo.regression.sgd.linear.LinearSGDTrainer;
import org.tribuo.regression.sgd.objectives.SquaredLoss;
import org.tribuo.regression.xgboost.XGBoostRegressionTrainer;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;


/**
 * Hello world with Tribuo ML!
 * @See https://github.com/oracle/tribuo/blob/main/docs/PackageOverview.md
 */


public class InsuranceCostPredictorApp
{
    
    public static void main(String[] args ) throws IOException {


        var regressionFactory = new RegressionFactory();
        var csvLoader = new CSVLoader<>(regressionFactory);

        //Load data
        String[] insuranceHeaders = new String[]{"Age", "Gender", "BodyMassIndex", "Children", "Smoke", "Department", "PremiumInsurance"};
        DataSource<Regressor> insuranceDataSource = csvLoader.loadDataSource(Paths.get("src/main/resources/insurance.csv"), "PremiumInsurance", insuranceHeaders);


        //split train and test data
        var splitter = new TrainTestSplitter<>(insuranceDataSource, 0.95, 0L);
        Dataset<Regressor> trainingDataset = new MutableDataset<>(splitter.getTrain());
        Dataset<Regressor> testingDataset = new MutableDataset<>(splitter.getTest());


        // Train the model on training Set
        // Option 1 LibLinearRegressionTrainer trainer = new LibLinearRegressionTrainer();
        //Option2 XGB var trainer = new XGBoostRegressionTrainer(25);

        LinearSGDTrainer trainer = new LinearSGDTrainer(new SquaredLoss(), new AdaGrad(0.5), 5, 77);
        Model<Regressor> model = trainer.train(trainingDataset);
        
        //Evaluate Model on test DataSet
        RegressionEvaluator evaluator = new RegressionEvaluator();
        Regressor dimension0 = new Regressor("DIM-0", Double.NaN);
        RegressionEvaluation score = evaluator.evaluate(model, testingDataset);
        // Metrics
        System.out.println("Testing Data Set");
        System.out.println("Root Mean Squared Error :" + score.rmse(dimension0));
        System.out.println("R-squared:" + Math.abs(score.r2(dimension0)));

        //predict Test values
       //ArrayExample(numFeatures=6,output=(DIM-0,11881.9696),weight=1.0,features=[(Age, 55.0)(BodyMassIndex, 30.14), (Children, 2.0), (Department, 1.0), (Gender, 0.0), (Smoke, 0.0), ])

        Regressor outputPlaceHolder = RegressionFactory.UNKNOWN_REGRESSOR;
        String[] featureNames = {"Age","Gender","BodyMassIndex","Children","Smoke","Department"};
        double[] featureValues = new double[]{25, 1.0,30,0,0,3};

        Example<Regressor> sample = new ArrayExample<>(outputPlaceHolder,featureNames,featureValues);
        Prediction<Regressor> prediction = model.predict(sample);
        double result = prediction.getOutput().getValues()[0];

        System.out.println("Predicted price =>"+result);

    }
    
}


