
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.*;
import java.util.Arrays;
import java.util.Random;
import java.util.StringTokenizer;
import java.util.Vector;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class RFhadoop extends Configured implements Tool{
	
	public static void main(String [] arg) throws Exception {
		
		int re = ToolRunner.run(new RFhadoop(), arg);
		System.exit(re);
	}
	
	public int run (String[] arg) throws Exception{
		Configuration conf = new Configuration();
		String path = "/tmp/train.csv";
		Path filePath = new Path(path);
		String uriWithLink = filePath.toUri().toString() + "#" + "train.csv";
		String path2 = "/tmp/test.csv";
		Path filepath2 = new Path(path2);
		String uriWithLink2 = filepath2.toUri().toString() + "#" + "test.csv";
		DistributedCache.addCacheFile(new URI(uriWithLink), conf);
		DistributedCache.addCacheFile(new URI(uriWithLink2), conf);
		Job job = new Job(conf);
		
		job.setJarByClass(RFhadoop.class);
		job.setJobName("rf");
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(Text.class);
		
		job.setMapperClass(RFMap.class);
		job.setReducerClass(RFReduce.class);
		
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		
		

		FileInputFormat.setInputPaths(job, new Path(arg[0]));
		FileOutputFormat.setOutputPath(job, new Path(arg[1]));
		
		
		boolean success = job.waitForCompletion(true);
		return success ? 0:1;
		
	}
	
	
public static class RFMap extends Mapper<LongWritable, Text, LongWritable,  Text> {
 
	@Override
	protected void cleanup(
			Mapper<LongWritable, Text, LongWritable, Text>.Context context)
			throws IOException, InterruptedException {
		// TODO Auto-generated method stub
		super.cleanup(context);
	}

	private final static int TRAIN_SIZE = 5000;
	private final static int TEST_SIZE = 40000;
	private final static int SAMPLE_SIZE = 2500;
	private final static int FEATURE = 1024;
	private final static int CANDIDATE_FEATURE = 32;
	
	private final static int CLASS_NUMBER = 10;
	private final static int NODE_SIZE = 80;
	private final static double THRESHOLD = 0.1;
	private int trainSet[][] = new int[TRAIN_SIZE][FEATURE];
	private int testSet[][] = new int [TEST_SIZE][FEATURE];
	private int sampleTrainSet[][] = new int[SAMPLE_SIZE][CANDIDATE_FEATURE];
	private int value[] = new int[TRAIN_SIZE];
	private int sampleValue[] = new int[SAMPLE_SIZE];
	private int r[] = new int[FEATURE];
	private int g[] = new int[FEATURE];  
	private int b[] = new int[FEATURE];
	private int featureValue[] = new int[CANDIDATE_FEATURE];
	private Vector<RecordTreeNode> treeData = new Vector<RecordTreeNode>();
	private int predict[] = new int[TEST_SIZE];
	
	private void rgb2grayTrain(int k) {
		for (int i = 0; i < FEATURE; i++) 
			trainSet[k][i] = (r[i]*19595 + g[i]*38469 + b[i]*7472) >> 16; 
	}
	
	private void rgb2grayTest(int k) {
		for (int i = 0; i < FEATURE; i++) {
			testSet[k][i] = (r[i]*19595 + g[i]*38469 + b[i]*7472) >> 16; 
		}
	}
	
	@SuppressWarnings("resource")
	private void getTrainSet(Configuration conf) throws FileNotFoundException,  IOException {
		
		Path[]cacheFiles = DistributedCache.getLocalCacheFiles(conf);
		
		if (cacheFiles != null) {
			FileReader csv = new FileReader(cacheFiles[0].toString());
					
			BufferedReader buf = new BufferedReader(csv);
			buf.readLine();
			String line = "";
			int i = 0;
			while ((line = buf.readLine()) != null && i < TRAIN_SIZE) {
				StringTokenizer st = new StringTokenizer(line, ",");
				st.nextToken();
				value[i] = Integer.parseInt(st.nextToken());
				int j = 0; 
				while (st.hasMoreElements()) {
					if (j < FEATURE) 
						r [j] = Integer.parseInt(st.nextToken());
					else if (j < 2*FEATURE)
						g[j-FEATURE] = Integer.parseInt(st.nextToken());
					else
						b[j-2*FEATURE] = Integer.parseInt(st.nextToken());
					j++;
				}
				rgb2grayTrain(i);
				i++;
			
			}
		}
		
	}
	
	@SuppressWarnings("resource")
	private void getTestSet(Configuration conf) throws IOException {
		Path[]cacheFiles = DistributedCache.getLocalCacheFiles(conf);
		if (cacheFiles != null) {
			FileReader csv = new FileReader(cacheFiles[1].toString());
			
			BufferedReader buf = new BufferedReader(csv);
			buf.readLine();
			String line = "";
			int i = 0;
			while ((line = buf.readLine()) != null && i < TEST_SIZE) {
				StringTokenizer st = new StringTokenizer(line, ",");
				st.nextToken();
				
				int j = 0; 
				while (st.hasMoreElements()) {
					if (j < FEATURE) 
						r [j] = Integer.parseInt(st.nextToken());
					else if (j < 2*FEATURE)
						g[j-FEATURE] = Integer.parseInt(st.nextToken());
					else
						b[j-2*FEATURE] = Integer.parseInt(st.nextToken());
					j++;
				}
				rgb2grayTest(i);
				i++;
			}
		}
	}
	
	
	@Override
	protected void setup(
			Mapper<LongWritable, Text, LongWritable, Text>.Context context)
			throws IOException, InterruptedException {

		Configuration conf = context.getConfiguration();
		getTrainSet(conf);
		getTestSet(conf);
		
	}
	
	@Override
	protected void map(LongWritable key, Text val,
			Mapper<LongWritable, Text, LongWritable, Text>.Context context)
			throws IOException, InterruptedException {
		// TODO Auto-generated method stub
		//super.map(key, value, context);
		int sample[] = new int[SAMPLE_SIZE];
		int tempRecordposition[] = new int[SAMPLE_SIZE];
		
		for (int j = 0; j < SAMPLE_SIZE; j++) {
			sample[j] = new Random().nextInt(TRAIN_SIZE);
		}
		
		for (int j = 0; j < CANDIDATE_FEATURE; j++) {
			featureValue[j] = new Random().nextInt(FEATURE);
		}
		
		for (int i = 0; i < SAMPLE_SIZE; i++) {
			for (int j = 0; j < CANDIDATE_FEATURE; j++) {
				sampleTrainSet[i][j] = trainSet[sample[i]][featureValue[j]];
			}
			sampleValue[i] = value[sample[i]];
			tempRecordposition[i] = i;
		}
		Long begain = System.currentTimeMillis();
		createDecisionTree(tempRecordposition,SAMPLE_SIZE);

		Long end = System.currentTimeMillis();
		System.out.println(val.toString() + " " + (end-begain));
		
		doValidation();
		
		
		String str = key.toString();
		for (int i = 0; i < TEST_SIZE; i++) {
			str += "," + Integer.toString(predict[i]); 
		}
		
		LongWritable newKey = new LongWritable(1);
		treeData.clear();
		//System.out.println(str);
		context.write(newKey, new Text(str));
	}
	
	public double log(double base, double val) {
		return Math.log(val)/Math.log(base);
	}
	
	private int[] getEntropyArray(int myRecord[], int si) {
		int temp[][] = new int[si][CLASS_NUMBER];
		int temp2[] = new int[CLASS_NUMBER];
		for (int i = 0; i < si; i++) {
			int c = sampleValue[myRecord[i]];
			for (int j = 0; j < CLASS_NUMBER; j++) {
				if (i != 0)
					temp[i][j] = temp[i-1][j];
				else
					temp[i][j] = 0;
			}
			temp[i][c]++;
			temp2[c] = temp[i][c];
		}
		return temp2;
	}
	
	private int getClassValue(int myRecord[], int si) {
		
		int classFrequency[] = new int[CLASS_NUMBER];
		for (int i = 0; i < si; i++) {
			int c = sampleValue[myRecord[i]];
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
		
		treeData.add(recordFeatureValue);
		return treeData.size() - 1;
		
	}
	

	public int createDecisionTree(int tempRecordPosition[], int size) {
	
		int judgeArray[] = getEntropyArray(tempRecordPosition,size);
		double judgeValue = 0.;
		for (int i = 0; i < CLASS_NUMBER; i++) {
			double temp = 1.0*judgeArray[i]/size;
			if (judgeArray[i] > 0)
				judgeValue -= temp*log(temp,2);
		}
		if (judgeValue < THRESHOLD  || size < NODE_SIZE) {	
			int t = getClassValue(tempRecordPosition, size);
			return newNode(-1, -1, t, -1,-1);
		}
		
		double minEntropy = 1000;
		int indexSplitFeature = -1;
		int indexSplitValue = -1;
		int indexSplitIndex = 0;
		Tuple sampleTuples[] = new Tuple[size];
		
		for (int i = 0; i < size; i++) {
			sampleTuples[i] = new Tuple();
			sampleTuples[i].row = tempRecordPosition[i];
		}
	
		for (int i = 0; i < CANDIDATE_FEATURE; i++) {
			
			//sort
			for (int j = 0; j < size; j++) {
				sampleTuples[j].col = i;
			}
			
			System.setProperty("java.util.Arrays.useLegacyMergeSort", "true");
			Arrays.sort(sampleTuples);
			
			//get class number
			int indexClass[][] = new int[size][CLASS_NUMBER];
			for (int j = 0; j < size; j++) {
				int c = sampleValue[sampleTuples[j].row];

				
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
				if (sampleTrainSet[row][i] != sampleTrainSet[row2][i]) {
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
						minEntropy = totalEntropy;
						indexSplitFeature = i;
						indexSplitValue = sampleTrainSet[row][i];
						indexSplitIndex = j;
					}
				}
			}
		}
		Tuple tempTuples[] = new Tuple[size];
		int tempFeature = indexSplitFeature == -1 ? -1:featureValue[indexSplitFeature];
		for (int i = 0; i < size; i++) {
			tempTuples[i] = new Tuple();
			tempTuples[i].row = tempRecordPosition[i];
			tempTuples[i].col = indexSplitFeature;
		}
		
		System.setProperty("java.util.Arrays.useLegacyMergeSort", "true");
		if (tempTuples[0].col != -1)
			Arrays.sort(tempTuples);
		int recordPositionLeft[] = new int[indexSplitIndex+1];
		int recordPositionRight[] = new int[size-indexSplitIndex-1];
		for (int i = 0; i < indexSplitIndex + 1; i++) {
			recordPositionLeft[i] = tempTuples[i].row;
		}
		for (int i = indexSplitIndex + 1; i < size; i++) {
			recordPositionRight[i - indexSplitIndex - 1] = tempTuples[i].row;
		}
		
		int lchild = createDecisionTree(recordPositionLeft, indexSplitIndex+1);
		int rchild = createDecisionTree(recordPositionRight,size - indexSplitIndex -1);
		
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
				
				return sampleTrainSet[this.row][this.col] == sampleTrainSet[t.row][t.col] ? 0 :
					(sampleTrainSet[this.row][this.col] > sampleTrainSet[t.row][t.col] ? 1: -1);
			}
			else
				return 0;
		}
	}

	
	private void doValidation() {
		
		for (int i = 0; i < TEST_SIZE; i++) {
			int index = treeData.size() - 1;
			RecordTreeNode node = treeData.get(index);
			
			while (true) {
				if (node.feature == -1) {
					predict[i] = node.prediction;
					break;
				}
				if (node.value >= testSet[i][node.feature]) 
					index = node.leftChild;
				else 
					index = node.rightChild;
				node  = treeData.get(index);
			}
		}
	}
	
}

public static class RFReduce extends Reducer<LongWritable, Text, LongWritable, Text> {

	private final static int TEST_SIZE = 40000;
	private final static int CLASS_NUMBER = 10;
	private int predictValue[][] = new int[TEST_SIZE][CLASS_NUMBER];
	private int finalPredict[] = new int[TEST_SIZE];
	
	@Override
	protected void reduce(LongWritable arg0, Iterable<Text> arg1,
			Reducer<LongWritable, Text, LongWritable, Text>.Context arg2)
			throws IOException, InterruptedException {
		List<String>value = new ArrayList<String>();
		for (Text t : arg1) {
			value.add(t.toString());
		}
		
		for (String line : value) {
			
			StringTokenizer st = new StringTokenizer(line.toString(), ",");
			int i = 0;
			st.nextToken();
			while (st.hasMoreTokens()) {
				int classValue = Integer.parseInt(st.nextToken());
				predictValue[i][classValue]++;
				i++;
			}
		}
		for (int i = 0; i < TEST_SIZE; i++) {
			int max = 0;
			for (int j = 0; j < CLASS_NUMBER; j++) {
				if (max < predictValue[i][j]) {
					max = predictValue[i][j];
					finalPredict[i] = j;
				}
			}
		}
		String result = "";
		for (int i = 0; i < TEST_SIZE; i++) {
			result += Integer.toString(finalPredict[i]) + ",";
		}
		arg2.write(arg0, new Text(result));
	}
	
	
	
}
	
}
