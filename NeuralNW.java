import java.util.*;
import java.io.*;
import java.lang.*;
import java.math.*;

public class NeuralNW{

	static int inputNodes = 4;
	static int hiddenNodes = 32;
	static double learnRate = 0.05;
	static double maxError = 0.05;

	static int epochs = 0;

	static double inpHidden[][] = new double[inputNodes][hiddenNodes];
	static double outHidden[] = new double[hiddenNodes];

	static double input[] = new double[inputNodes];
	static double output[] = new double[1];
	static double hidden[] = new double[hiddenNodes];

	static double inpIn[] = new double[inputNodes];
	static double outIn[] = new double[1];
	static double hiddenIn[] = new double[hiddenNodes];

	static double inpDelta[] = new double[inputNodes];
	static double outDelta[] = new double[1];
	static double hiddenDelta[] = new double[hiddenNodes];

	static double sigmoid(double v){
		return 1.0 / (1.0 + Math.exp(-v));
	}

	static double sigmoidDeriv(double v){
		return Math.exp(-v) / Math.pow(1.0 + Math.exp(-v), 2);
	}

	static void exeLearning(int[][] examples){
		for(int i=0; i<inputNodes; i++){
			for(int j=0; j<hiddenNodes; j++){
				inpHidden[i][j] = Math.random() * (1 + 1) - 1;
				//inpHidden[i][j] = random.nextInt(1 + 1 + 1) - 1;
				//System.out.println("RANDOM: " + inpHidden[i][j]);
			}
		}

		for(int i=0; i<hiddenNodes; i++){
			outHidden[i] = Math.random() * (1 + 1) - 1;
		}

		boolean halt;

		do{
			halt = true;
			epochs++;

			for(int i=0; i<examples.length; i++){
				//System.out.println("LENGTH: " + examples[i].length);
				for(int j=0; j<examples[i].length-1; j++){
					input[j] = examples[i][j];
				}

				for(int j=0; j<hiddenNodes; j++){
					hiddenIn[j] = 0;

					for(int k=0; k<inputNodes; k++){
						hiddenIn[j] += inpHidden[k][j] * input[k];
					}
					hidden[j] = sigmoid(hiddenIn[j]);
				}
				outIn[0] = 0;

				for(int k=0; k<hiddenNodes; k++){
					outIn[0] += outHidden[k] * hidden[k];
				}
				output[0] = sigmoid(outIn[0]);

				outDelta[0] = sigmoidDeriv(outIn[0]) * (examples[i][examples[i].length-1] - output[0]);

				for(int k=0; k<hiddenNodes; k++){
					hiddenDelta[k] = sigmoidDeriv(hiddenIn[k]) * outHidden[k] * outDelta[0];
				}

				for(int k=0; k<inputNodes; k++){
					inpDelta[k] = 0;

					for(int j=0; j<hiddenNodes; j++){
						inpDelta[k] += inpHidden[k][j] * hiddenDelta[j];
					}
					inpDelta[k] *= sigmoidDeriv(input[k]);
				}

				for(int k=0; k<inputNodes; k++){

					for(int j=0; j<hiddenNodes; j++){
						inpHidden[k][j] += learnRate * input[k] * hiddenDelta[j];
					}
				}

				for(int k=0; k<hiddenNodes; k++){
					outHidden[k] += learnRate * hidden[k] * outDelta[0];
				}

				if(Math.abs(examples[i][examples[i].length-1] - output[0]) > maxError){
					/*
					System.out.println("EXAMPLES NOT YET LEARNED:");

					for(int j=0; j<inputNodes; j++){
						System.out.print(examples[i][j] + " <-> ");
					}
					
					System.out.println("EXPECTED: " + examples[i][examples[i].length-1]);
					System.out.println("GOT: " + output[0]);
					System.out.println("ERROR: " + Math.abs(examples[i][examples[i].length-1] - output[0]));
					*/
					halt = false;
				}
			}
		} while(!halt);

		System.out.println("NUMBER OF EPOCHS: " + epochs);
	}

	static double exeTesting(int test[]){

		for(int i=0; i<inputNodes; i++){
			input[i] = test[i];
		}

		for(int j=0; j<hiddenNodes; j++){
			hiddenIn[j] = 0;

			for(int k=0; k<inputNodes; k++){
				hiddenIn[j] += inpHidden[k][j] * input[k];
			}
			hidden[j] = sigmoid(hiddenIn[j]);
		}
		outIn[0] = 0;

		for(int k=0; k<hiddenNodes; k++){
			outIn[0] += outHidden[k] * hidden[k];
		}
		output[0] = sigmoid(outIn[0]);

		return output[0];
	}

	public static void main(String[] args){

		Scanner in = new Scanner(System.in);

		if(args.length != 1){
			System.out.println("USAGE: NeuralNW <EXAMPLES_FILE>");
			System.exit(1);
		}

		String file = args[0];
		String line;
		BufferedReader name = null;
		int rows = 0;
		int cols = 0;
		int data[][] = new int[300][300];

		try{
			name = new BufferedReader(new FileReader(file));
			while((line = name.readLine()) != null){
				String[] nums = line.split(" ");

				for(int i=0; i<nums.length; i++){
					try{
						data[rows][i] = Integer.parseInt(nums[i]);
					}
					catch (NumberFormatException e){
						data[rows][i] = 0;
					}
				}
				rows++;
				cols = nums.length;
			}
		} catch(FileNotFoundException e){
      		e.printStackTrace();
    	} catch(IOException e){
     	 	e.printStackTrace(); 
		} finally{
			if(name != null){
				try{
					name.close();
				} catch(IOException e){
					e.printStackTrace();
				}
			}
		}
		//System.out.println("ROWS: " + rows + "\nCOLS: " + cols);

		int dataFinal[][] = new int[rows][cols];
		for(int i=0; i<rows; i++){
			for(int j=0; j<cols; j++){
				dataFinal[i][j] = data[i][j];
			}
		}

		exeLearning(dataFinal);
		int inpArr[] = new int[inputNodes];
		System.out.print("NUMBER OF TESTS TO MAKE: ");
		int nTests = in.nextInt();

		for(int i=0; i<nTests; i++){
			System.out.print("INPUT: ");

			for(int j=0; j<inputNodes; j++){
				inpArr[j] = in.nextInt();
			}

			double result = exeTesting(inpArr);
			System.out.println("OUTPUT: " + result);
		}
	}
}