import java.util.*;
import java.io.*;
public class RandomForest {

	private final static int FEATURE = 1024;
	private final static int TESTSET = 40000;
	private final static int TRAINSET = 20000;
	private final static int TREE_NUMBER = 200;
	private final static int CANDIDATE_FEATURE = 32;
	private final static int SAMPLE_TUPLE = 10000;
	private final static int CLASS_NUMBER = 10;
	private final static int NODE_SIZE = 50;
	private final static double THRESHOLD = 0.1;
	private int depth = 0;
	private int r[] = new int[FEATURE];
	private int g[] = new int[FEATURE];
	private int b[] = new int[FEATURE];
	private int value[] = new int[TRAINSET];
	private int trainSet[][] = new int[TRAINSET][FEATURE];
	private int testSet[][] = new int[TESTSET][FEATURE];
	private int predictValue[] = new int[TESTSET];
	private int sampleTrain[] = new int[SAMPLE_TUPLE];
	private int featureValue[] = new int[CANDIDATE_FEATURE];
	
	/*
	private int newTestSet[][] = new int[5000][FEATURE];
	private int NEWTRAINSET = 15000;
	private int NEWTESTSET = 5000;
	*/
	//private int recordSplitFeature[][] = new int [2][CANDIDATE_FEATURE];
	private Vector<RecordTreeNode> data = new Vector<RecordTreeNode>();
	public Vector<Vector<RecordTreeNode>> forest = new Vector<Vector<RecordTreeNode>>();
	
	
	private void trainRgbToGray(int k) {
		for (int i = 0; i < FEATURE; i++) {
			trainSet[k][i] = (r[i]*19595 + g[i]*38469 + b[i]*7472) >> 16;
		}
		
	}
	
	private void testRgbToGray(int k) {
		for (int i = 0; i < FEATURE; i++) {
			testSet[k][i] = (r[i]*19595 + g[i]*38469 + b[i]*7472) >> 16;
		}
	}
	
	public void getTrainData() throws FileNotFoundException,IOException{
		File csv = new File("train.csv");
		BufferedReader buf = new BufferedReader(new FileReader(csv));
		buf.readLine();
		
		String line = "";
		int i = 0;
		while ((line = buf.readLine()) != null) {
			StringTokenizer st = new StringTokenizer(line,",");
			st.nextToken();
			value[i] = Integer.parseInt(st.nextToken());
			int j = 0;
			while (st.hasMoreTokens()) {
				if (j < FEATURE)
					r[j] = Integer.parseInt(st.nextToken());
				else if (j < 2*FEATURE) 
					g[j-FEATURE] = Integer.parseInt(st.nextToken());
				else
					b[j-2*FEATURE] = Integer.parseInt(st.nextToken());
				j++;
			}
			trainRgbToGray(i);
			i++;
		}
		buf.close();
	}
	
	public void getTestData() throws FileNotFoundException,IOException {
		File csv = new File("test.csv");
		BufferedReader buf = new BufferedReader(new FileReader(csv));
		buf.readLine();
		String line = "";
		int i = 0;
		while( (line = buf.readLine()) != null) {
			StringTokenizer st = new StringTokenizer(line,",");
			st.nextToken();
			int j = 0;
			while(st.hasMoreTokens()) {
				
				if (j < FEATURE)
					r[j] = Integer.parseInt(st.nextToken());
				else if (j < FEATURE*2)
					g[j-FEATURE] = Integer.parseInt(st.nextToken());
				else
					b[j - 2*FEATURE] = Integer.parseInt(st.nextToken());
				j++;
			}
			testRgbToGray(i);
			i++;
		}
		buf.close();
	}
	
	
		
	@SuppressWarnings("unchecked")
	public void buildDecisionTree() {
		
		//
		for (int i = 0; i < SAMPLE_TUPLE; i++) {
			sampleTrain[i] = new Random().nextInt(TRAINSET);
		}
		for (int i = 0; i < CANDIDATE_FEATURE; i++) {
			featureValue[i] = new Random().nextInt(FEATURE);
		}
		/*
		for (int i = 0; i < CANDIDATE_FEATURE; i++) {
			recordSplitFeature[0][i] = 0;
			recordSplitFeature[1][i] = 0;
		}
		*/
		depth = 0;
		createDecisionTree(sampleTrain,0,SAMPLE_TUPLE,depth);

		Vector<RecordTreeNode> tempNode = new Vector<RecordTreeNode>();
		tempNode = (Vector<RecordTreeNode>) data.clone();
		forest.addElement(tempNode);
		
		data.clear();
	}
	
	public static double log(double value, double base) {
		return Math.log(value) / Math.log(base);
	}
	
	private int[] getEntropyArray(int myRecord[], int l, int r) {
		int temp[][] = new int[r-l][CLASS_NUMBER];
		int temp2[] = new int[CLASS_NUMBER];
		for (int i = l; i < r; i++) {
			int c = value[myRecord[i-l]];
			for (int j = 0; j < CLASS_NUMBER; j++) {
				if (i != l)
					temp[i-l][j] = temp[i-1-l][j];
				else
					temp[i-l][j] = 0;
			}
			temp[i-l][c]++;
			temp2[c] = temp[i-l][c];
		}
		return temp2;
	}
	
	private int getClassValue(int myRecord[], int le, int r) {
		
		int classFrequency[] = new int[CLASS_NUMBER];
		for (int i = le; i < r; i++) {
			int c = value[myRecord[i-le]];
			for (int j = 0; j < CLASS_NUMBER; j++) {
				if (j == c)
					classFrequency[c]++;
			}
		}
		int classValue = -1;
		int maxFrequency = 0;
		for (int i = 0; i < CLASS_NUMBER; i++) {
			if (maxFrequency < classFrequency[i]) {
				maxFrequency = classFrequency[i];
				classValue = i;
			}
		}
		return classValue;
	}
	
	private int newNode (int fe, int va, int pre, int lc, int rc) {
		RecordTreeNode recordFeatureValue = new RecordTreeNode(fe, va, pre, lc, rc);
		
		data.add(recordFeatureValue);
		//System.out.print( data.size() - 1);
		return data.size() - 1;
		
	}
	
	public static void display(Tuple[] tuple){  
        for (int i = 0; i < tuple.length; i++) {  
            System.out.println(tuple[i].row + "		" + tuple[i].col);  
        }  
    }
	

	public int createDecisionTree(int tempRecordPosition[], int left, int right, int dep) {
		int size = right - left;
		depth = dep + 1;
	
		int judgeArray[] = getEntropyArray(tempRecordPosition,left,right);
		double judgeValue = 0.;
		for (int i = 0; i < CLASS_NUMBER; i++) {
			double temp = 1.0*judgeArray[i]/size;
			if (judgeArray[i] > 0)
				judgeValue -= temp*log(temp,2);
		}
		//if (judgeValue < THRESHOLD || (signalLeft == 0 && si == 0) || (si == 1 && signalRight == 0)) {
		if (judgeValue < THRESHOLD  || size < NODE_SIZE) {	
			int t = getClassValue(tempRecordPosition, left, right);
			return newNode(-1, -1, t, -1,-1);
		}
		
		double minEntropy = 1000;
		int indexSplitFeature = -1;
		int indexSplitValue = -1;
		int indexSplitIndex = 0;
		Tuple sampleTuples[] = new Tuple[size];
		
		for (int i = left; i < right; i++) {
			sampleTuples[i-left] = new Tuple();
			sampleTuples[i-left].row = tempRecordPosition[i-left];
		}
	
		for (int i = 0; i < CANDIDATE_FEATURE; i++) {
			
			int index = featureValue[i];
			
			//sort
			for (int j = 0; j < size; j++) {
				sampleTuples[j].col = index;
			}
			System.setProperty("java.util.Arrays.useLegacyMergeSort", "true");
			Arrays.sort(sampleTuples);
			
			//get class number
			int indexClass[][] = new int[size][CLASS_NUMBER];
			for (int j = 0; j < size; j++) {
				int c = value[sampleTuples[j].row];

				
				for (int k = 0; k < CLASS_NUMBER; k++) {
					if (j > 0)
						indexClass[j][k] = indexClass[j-1][k];
					else
						indexClass[j][k] = 0;
				}
				indexClass[j][c]++;
			}
			
			//get entropy
			for (int j = 0; j < size - 1; j++) {
				
				int row = sampleTuples[j].row;
				int row2 = sampleTuples[j+1].row;
				//System.out.println(trainSet[row][index] + "hh" + trainSet[row2][index] + "index " + index );
				if (trainSet[row][index] != trainSet[row2][index]) {
					//System.out.println("gg");
					double entropy1 = 0, entropy2 = 0;
					for (int k = 0; k < CLASS_NUMBER; k++) {
						int v1 = indexClass[j][k], v2 = indexClass[size-1][k] - indexClass[j][k];
						if (v1 != 0) {
							double p1 = 1.0*v1/(j+1);
							entropy1 -= p1*log(p1,2);
						}
						 
						if (v2 != 0) {
							double p2 = 1.0*v2/(size-j-1);
							entropy2 -= p2*log(p2,2);
						}
					}
					
					double totalEntropy = entropy1 * (j+1) / size + entropy2 * (size-j-1) / size;
					
					if (totalEntropy < minEntropy) {
						//System.out.println("helloc " + totalEntropy + " " + featureValue[i]);
						minEntropy = totalEntropy;
						indexSplitFeature = i;
						//System.out.println("row: " + sampleTuples[j].row  + "col " + sampleTuples[j].col);
						indexSplitValue = trainSet[sampleTuples[j].row][sampleTuples[j].col];
						indexSplitIndex = j;
					}
				}
			}
		}
		Tuple tempTuples[] = new Tuple[size];
		for (int i = left; i < right; i++) {
			tempTuples[i-left] = new Tuple();
			tempTuples[i-left].row = tempRecordPosition[i-left];
		}
		int tempFeature = -1;
		if (indexSplitFeature != -1) {
			for (int j = 0; j < size; j++) {
				tempFeature = featureValue[indexSplitFeature];
				tempTuples[j].col = tempFeature;		
			}
			/*
			if (si == 0)
				recordSplitFeature[0][indexSplitFeature] = 1;
			else
				recordSplitFeature[1][indexSplitFeature] = 1;
				*/
		}
		else {
			for (int j = 0; j < size; j++) {
				tempTuples[j].col = -1;		
			}
		}
	
			
		//display(sampleTuples);
		System.setProperty("java.util.Arrays.useLegacyMergeSort", "true");
		if (tempTuples.length > 1)
			Arrays.sort(tempTuples);
		int recordPositionLeft[] = new int[indexSplitIndex+1];
		int recordPositionRight[] = new int[size-indexSplitIndex-1];
		for (int i = 0; i < indexSplitIndex + 1; i++) {
			recordPositionLeft[i] = tempTuples[i].row;
		}
		for (int i = indexSplitIndex + 1; i < size; i++) {
			recordPositionRight[i - indexSplitIndex - 1] = tempTuples[i].row;
		}
		
		int lchild = createDecisionTree(recordPositionLeft, 0, indexSplitIndex+1,depth);
		int rchild = createDecisionTree(recordPositionRight,indexSplitIndex+1,size, depth);
		
		return newNode(tempFeature, indexSplitValue, -1, lchild, rchild);
	} 
	
	class RecordTreeNode {
		int feature;
		int value;
		int prediction;
		int leftChild;
		int rightChild;
		public RecordTreeNode(int f, int v,int p,int lc, int rc) {
			// TODO Auto-generated constructor stub
	
			feature = f;
			value = v;
			prediction = p;
			leftChild = lc;
			rightChild = rc;
		}
	}
	

	class Tuple implements Comparable<Tuple>{
		int col;
		int row;
	
		@Override
		public int compareTo(Tuple arg0) {
			// TODO Auto-generated method stub
			Tuple t = (Tuple) arg0;
			if (this.col != -1 && this.row != -1 && t.row != -1 && t.col != -1) {
				
				return trainSet[this.row][this.col] >= trainSet[t.row][t.col] ? 1: -1;
			}
			return 0;
		}
	}

	public void Validation() {
		//int tempTest[][] = new int[TRAINSET][FEATURE];
		//tempTest = trainSet;
		int tempPredictValue[][] = new int[TESTSET][CLASS_NUMBER];
		for (int i = 0; i < TESTSET; i++) {
			
			for (int j = 0; j < TREE_NUMBER; j++) {
				Vector<RecordTreeNode> tempTreeNode = new Vector<RecordTreeNode>();
				tempTreeNode = forest.get(j);
				//System.out.println(tempTreeNode.size());
				int tempClassValue = 0;
				int node = tempTreeNode.size() - 1;
				while (true) {
					RecordTreeNode tempNode = tempTreeNode.get(node);
					//System.out.println(tempNode.leftChild);
					if ( tempNode.feature == -1) {
						tempClassValue = tempNode.prediction;
						break;
					}
					if (testSet[i][tempNode.feature] <= tempNode.value)
						node = tempNode.leftChild;
					else
						node = tempNode.rightChild;
				}
				tempPredictValue[i][tempClassValue]++;
			}
			int max = 0;
			for (int j = 0; j < CLASS_NUMBER; j++) {
				if (max < tempPredictValue[i][j]) {
					max = tempPredictValue[i][j];
					predictValue[i] = j;
				}
			}
		}
		writeResult();

	}
	
	
	
	private void writeResult() {
		File csv = new File("result.csv");
		try {
			BufferedWriter bw = new BufferedWriter(new FileWriter(csv));
			bw.newLine();
			bw.write("id" + "," + "label");
			for (int i = 0; i < TESTSET; i++) {
				bw.newLine();
				bw.write(i + "," + predictValue[i]);
			}
			bw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

    public void run () {
    	for (int i = 0; i < TREE_NUMBER; i++) {
			buildDecisionTree();
			System.out.println("building Tree: " + i);
		} 
    	Validation();
    }
	
	public static void main(String []args) throws FileNotFoundException, IOException {
		RandomForest rf = new RandomForest();
		
		long startTime = System.currentTimeMillis();
		rf.getTrainData();
		rf.getTestData();
		long endTime = System.currentTimeMillis();
		System.out.println("FileTime: " + (endTime - startTime));
		//rf.start();
		long startTime3 = System.currentTimeMillis();
		rf.run();
		long endTime3 = System.currentTimeMillis();
		System.out.println(" Buildtime " + (endTime3 - startTime3));
		
		
		
	}
	

}
