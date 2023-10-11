package com.example;

import org.jpmml.evaluator.*;
import org.jpmml.model.PMMLUtil;
import org.xml.sax.SAXException;

import javax.xml.parsers.ParserConfigurationException;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class InsurancePMMLExample {


    public static void main(String []args ){


        // Building a model evaluator from a PMML file
        try {
            Evaluator evaluator = new LoadingModelEvaluatorBuilder()
                    .load(new File("src/main/resources/insurancePriceModel.pmml"))
                    .build();

            //  self-check
            evaluator.verify();

            // Printing Model input features
            List<InputField> inputFields = evaluator.getInputFields();
            System.out.println("Input Features: " + inputFields);

            // Printing Target variable
            List<TargetField> targetFields = evaluator.getTargetFields();
            System.out.println("Target Variable: " + targetFields);


            //Prediction
            // Create a prediction input
            Map<String, Object> input = new HashMap<>();
            input.put("Age", 25);
            input.put("Gender", 1.0);
            input.put("BodyMassIndex", 33);
            input.put("Childrens", 0);
            input.put("Smoke", 0);
            input.put("Department", 3);

            Object prediction = evaluator.evaluate(input);
            System.out.println("Predicted Insurance Premium price "+prediction);


        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }



    }
}
