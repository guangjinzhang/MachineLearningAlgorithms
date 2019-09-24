import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
public class NaiveBayes {
	//calculate the mean value of each feature	
	public static Double mean(List<Double> list){
		Double sum=0.0;
		for(Double l:list){
			sum+=l;
		}
		return sum/(double)list.size();
	}
	
	//calculate the variance based on the mean
	public static Double variance(List<Double> list){
		Double sum=0.0;
		for(Double l:list){
		sum+=Math.pow((l-mean(list)), 2);
		}
		return sum/(double)list.size();		
	}
	
	//calculate Guassian based on variance and mean for each feature	
	public static Double Guassian(Double value, List<Double> list){	
		double a=Math.exp(-Math.pow(value-mean(list), 2)/2*variance(list));
		//if the a is infinitesimal in Math.exp(a), Math.exp(a) will be 0.
		//This method is used for avoiding zero value of Guassion. 
		if (a==0){
			a=1E-10;
		}
		double b=Math.sqrt(2*Math.PI*variance(list));
		return a/b;
	}
	
	//get value list of each feature from all data
	public static List<Double> getList(Map<String,List<Double>> map,int index){
		List<Double> l=new ArrayList<Double>();
		Set<String> Key=map.keySet();
		for(String k:Key){
			l.add(map.get(k).get(index));
		}
		return l;
	}
	
	//combine test and train map, then calculate the probability by Guassian
	public static Map<String, List<Double>> getPmap (Map <String, List<Double>> test, Map<String, List<Double>> train){
		//probability map
		Map<String, List<Double>> Pmap=new LinkedHashMap<String, List<Double>>();
		//each feature list get the train value by getList method 
		List<Double> fixed_acidity=getList(train,0);
		List<Double> volatile_acidity=getList(train,1);
		List<Double> citric_acid=getList(train,2);
		List<Double> residual_sugar=getList(train,3);
		List<Double> chlorides=getList(train,4);
		List<Double> free_sulfur_dioxide=getList(train,5);
		List<Double> total_sulfur_dioxide=getList(train,6);
		List<Double> density=getList(train,7);
		List<Double> pH=getList(train,8);
		List<Double> sulphates=getList(train,9);
		List<Double> alcohol=getList(train,10);
		//loop for calculating the probability value of each feature and store them into Pmap			
		Set<String> testkey=test.keySet();
		for(String testk:testkey){
			List<Double> Plist=new ArrayList<Double>();
			List<Double> list=test.get(testk);					
			Plist.add(Guassian(list.get(0),fixed_acidity));
			Plist.add(Guassian(list.get(1),volatile_acidity));
			Plist.add(Guassian(list.get(2),citric_acid));	
			Plist.add(Guassian(list.get(3),residual_sugar));
			Plist.add(Guassian(list.get(4),chlorides));
			Plist.add(Guassian(list.get(5),free_sulfur_dioxide));
			Plist.add(Guassian(list.get(6),total_sulfur_dioxide));
			Plist.add(Guassian(list.get(7),density));
			Plist.add(Guassian(list.get(8),pH));
			Plist.add(Guassian(list.get(9),sulphates));
			Plist.add(Guassian(list.get(10),alcohol));
			Pmap.put(testk, Plist);
		}
		return Pmap;		
	}
	
	//main method to calculate the accuracy and confusion matrix
	public static void main(String[] args) throws NumberFormatException, IOException{
		 //each quality map to store the corresponding sample
		 Map<String,List<Double>> ThreeMap=new LinkedHashMap<String,List<Double>>(); 
		 Map<String,List<Double>> FourMap=new LinkedHashMap<String,List<Double>>(); 
		 Map<String,List<Double>> FiveMap=new LinkedHashMap<String,List<Double>>();
		 Map<String,List<Double>> SixMap=new LinkedHashMap<String,List<Double>>();
		 Map<String,List<Double>> SevenMap=new LinkedHashMap<String,List<Double>>();
		 Map<String,List<Double>> EightMap=new LinkedHashMap<String,List<Double>>(); 
		 //read the red wine file
		 File file = new File("src\\winequality-red.csv");
		 String s=null;
		 List<String> list=new ArrayList<String>();
		 BufferedReader br = new BufferedReader(new FileReader(file));
		 while( (s = br.readLine()) != null) {
			 list.add(s);
		 }
			br.close();	
		//remove the features name on the first row
		list.remove(0);
		//split each row with ";", get each value
		for (String l:list){
			String[]k=l.split(";");
			//change string to double
			List<Double> Dlist=new ArrayList<Double>();
			for (String i:k){
				Dlist.add(Double.valueOf(i));
			}
			//classify all samples in each group based on the value of quality (the last one of list)
			if (Dlist.get(Dlist.size()-1)==3){
				ThreeMap.put("Three"+(ThreeMap.size()+1),Dlist);					
			}
			if (Dlist.get(Dlist.size()-1)==4){
				FourMap.put("Four"+(FourMap.size()+1),Dlist);					
			}
			if (Dlist.get(Dlist.size()-1)==5){
				FiveMap.put("Five"+(FiveMap.size()+1),Dlist);					
			}
			if (Dlist.get(Dlist.size()-1)==6){
				SixMap.put("Six"+(SixMap.size()+1),Dlist);					
			}
			if (Dlist.get(Dlist.size()-1)==7){
				SevenMap.put("Seven"+(SevenMap.size()+1),Dlist);					
			}
			if (Dlist.get(Dlist.size()-1)==8){
				EightMap.put("Eight"+(EightMap.size()+1),Dlist);					
			}	
		}	
		 double wholeaccuracy=0.0;
		 double wholesum=0.0;
		 //use 10-fold CV to split the train and test samples 
		 for(int k=1;k<=10;k++){
			 double accuracy=0.0;
			 //train map is used to store train samples for each group
			 Map<String, List<Double>> testmap=new LinkedHashMap<String, List<Double>>();
			 Map<String, List<Double>> trainmap1=new LinkedHashMap<String, List<Double>>();
			 Map<String, List<Double>> trainmap2=new LinkedHashMap<String, List<Double>>();
			 Map<String, List<Double>> trainmap3=new LinkedHashMap<String, List<Double>>();
			 Map<String, List<Double>> trainmap4=new LinkedHashMap<String, List<Double>>();
			 Map<String, List<Double>> trainmap5=new LinkedHashMap<String, List<Double>>();
			 Map<String, List<Double>> trainmap6=new LinkedHashMap<String, List<Double>>();
			 trainmap1.putAll(ThreeMap);
			 trainmap2.putAll(FourMap);
			 trainmap3.putAll(FiveMap);
			 trainmap4.putAll(SixMap);
			 trainmap5.putAll(SevenMap);
			 trainmap6.putAll(EightMap);
			 //equally split each group into 90% train samples and 10% test samples for each group based on k time.
			 //firstly, select test samples and store in order, then remove them form the original map as train samples map
			 for(int i=(k-1)*(int)(ThreeMap.size()*0.1)+1;i<=k*(int)(ThreeMap.size()*0.1);i++){
				testmap.put("Three"+i, ThreeMap.get("Three"+i));
				trainmap1.remove("Three"+i);
			 }				 
			 for(int i=(k-1)*(int)(FourMap.size()*0.1)+1;i<=k*(int)(FourMap.size()*0.1);i++){
				testmap.put("Four"+i, FourMap.get("Four"+i));
				trainmap2.remove("Four"+i);
			 }				 
			 for(int i=(k-1)*(int)(FiveMap.size()*0.1)+1;i<=k*(int)(FiveMap.size()*0.1);i++){
				testmap.put("Five"+i, FiveMap.get("Five"+i));
				trainmap3.remove("Five"+i);
			 }				 			 
			 for(int i=(k-1)*(int)(SixMap.size()*0.1)+1;i<=k*(int)(SixMap.size()*0.1);i++){
				testmap.put("Six"+i, SixMap.get("Six"+i));
				trainmap4.remove("Six"+i);
			 }				 
			 for(int i=(k-1)*(int)(SevenMap.size()*0.1)+1;i<=k*(int)(SevenMap.size()*0.1);i++){
				testmap.put("Seven"+i, SevenMap.get("Seven"+i));
				trainmap5.remove("Seven"+i);
			 }
			 for(int i=(k-1)*(int)(EightMap.size()*0.1)+1;i<=k*(int)(EightMap.size()*0.1);i++){
				testmap.put("Eight"+i, EightMap.get("Eight"+i));
				trainmap6.remove("Eight"+i);
			 }
			 //probability map is used to store probability value for each group based on each feature		 
			 Map<String, List<Double>> Pmap1=getPmap(testmap,trainmap1);
			 Map<String, List<Double>> Pmap2=getPmap(testmap,trainmap2);
			 Map<String, List<Double>> Pmap3=getPmap(testmap,trainmap3);
			 Map<String, List<Double>> Pmap4=getPmap(testmap,trainmap4);
			 Map<String, List<Double>> Pmap5=getPmap(testmap,trainmap5);
			 Map<String, List<Double>> Pmap6=getPmap(testmap,trainmap6);
			 //quality (group) map is used to store the result of classification for confusion matrix at each time
			 List<Integer> three=new ArrayList<Integer>();
			 List<Integer> four=new ArrayList<Integer>();
			 List<Integer> five=new ArrayList<Integer>();
			 List<Integer> six=new ArrayList<Integer>();
			 List<Integer> seven=new ArrayList<Integer>();
			 List<Integer> eight=new ArrayList<Integer>();
			 //loop for initializing each list with 0
			 for (int i=0;i<6;i++){
				 three.add(0);
				 four.add(0);
				 five.add(0);
				 six.add(0);
				 seven.add(0);
				 eight.add(0);
			 }			 
			 //calculate the probability of each quality
			 //P(quality|features...)=P(feature1|quality)*P(feature2|quality)*...*P(feature11|quality)*P(quality)/(P(feature1|quality)+P(feature2|quality)+...+P(feature11|quality))
			 Set<String> key=Pmap1.keySet();
			 double sum=0.0;
			 for(String name:key){	
				 double p1=Pmap1.get(name).get(0)*Pmap1.get(name).get(1)*Pmap1.get(name).get(2)*Pmap1.get(name).get(3)*Pmap1.get(name).get(4)*Pmap1.get(name).get(5)*Pmap1.get(name).get(6)*Pmap1.get(name).get(7)*Pmap1.get(name).get(8)*Pmap1.get(name).get(9)*Pmap1.get(name).get(10)/
						 (Pmap1.get(name).get(0)+Pmap1.get(name).get(1)+Pmap1.get(name).get(2)+Pmap1.get(name).get(3)+Pmap1.get(name).get(4)+Pmap1.get(name).get(5)+Pmap1.get(name).get(6)+Pmap1.get(name).get(7)+Pmap1.get(name).get(8)+Pmap1.get(name).get(9)+Pmap1.get(name).get(10))*(double)ThreeMap.size()/(double)list.size();
				 double p2=Pmap2.get(name).get(0)*Pmap2.get(name).get(1)*Pmap2.get(name).get(2)*Pmap2.get(name).get(3)*Pmap2.get(name).get(4)*Pmap2.get(name).get(5)*Pmap2.get(name).get(6)*Pmap2.get(name).get(7)*Pmap2.get(name).get(8)*Pmap2.get(name).get(9)*Pmap2.get(name).get(10)/
						 (Pmap2.get(name).get(0)+Pmap2.get(name).get(1)+Pmap2.get(name).get(2)+Pmap2.get(name).get(3)+Pmap2.get(name).get(4)+Pmap2.get(name).get(5)+Pmap2.get(name).get(6)+Pmap2.get(name).get(7)+Pmap2.get(name).get(8)+Pmap2.get(name).get(9)+Pmap2.get(name).get(10))*(double)FourMap.size()/(double)list.size();
				 double p3=Pmap3.get(name).get(0)*Pmap3.get(name).get(1)*Pmap3.get(name).get(2)*Pmap3.get(name).get(3)*Pmap3.get(name).get(4)*Pmap3.get(name).get(5)*Pmap3.get(name).get(6)*Pmap3.get(name).get(7)*Pmap3.get(name).get(8)*Pmap3.get(name).get(9)*Pmap3.get(name).get(10)/
						 (Pmap3.get(name).get(0)+Pmap3.get(name).get(1)+Pmap3.get(name).get(2)+Pmap3.get(name).get(3)+Pmap3.get(name).get(4)+Pmap3.get(name).get(5)+Pmap3.get(name).get(6)+Pmap3.get(name).get(7)+Pmap3.get(name).get(8)+Pmap3.get(name).get(9)+Pmap3.get(name).get(10))*(double)FiveMap.size()/(double)list.size();
				 double p4=Pmap4.get(name).get(0)*Pmap4.get(name).get(1)*Pmap4.get(name).get(2)*Pmap4.get(name).get(3)*Pmap4.get(name).get(4)*Pmap4.get(name).get(5)*Pmap4.get(name).get(6)*Pmap4.get(name).get(7)*Pmap4.get(name).get(8)*Pmap4.get(name).get(9)*Pmap4.get(name).get(10)/
						 (Pmap4.get(name).get(0)+Pmap4.get(name).get(1)+Pmap4.get(name).get(2)+Pmap4.get(name).get(3)+Pmap4.get(name).get(4)+Pmap4.get(name).get(5)+Pmap4.get(name).get(6)+Pmap4.get(name).get(7)+Pmap4.get(name).get(8)+Pmap4.get(name).get(9)+Pmap4.get(name).get(10))*(double)SixMap.size()/(double)list.size();
				 double p5=Pmap5.get(name).get(0)*Pmap5.get(name).get(1)*Pmap5.get(name).get(2)*Pmap5.get(name).get(3)*Pmap5.get(name).get(4)*Pmap5.get(name).get(5)*Pmap5.get(name).get(6)*Pmap5.get(name).get(7)*Pmap5.get(name).get(8)*Pmap5.get(name).get(9)*Pmap5.get(name).get(10)/
						 (Pmap5.get(name).get(0)+Pmap5.get(name).get(1)+Pmap5.get(name).get(2)+Pmap5.get(name).get(3)+Pmap5.get(name).get(4)+Pmap5.get(name).get(5)+Pmap5.get(name).get(6)+Pmap5.get(name).get(7)+Pmap5.get(name).get(8)+Pmap5.get(name).get(9)+Pmap5.get(name).get(10))*(double)SevenMap.size()/(double)list.size();
				 double p6=Pmap6.get(name).get(0)*Pmap6.get(name).get(1)*Pmap6.get(name).get(2)*Pmap6.get(name).get(3)*Pmap6.get(name).get(4)*Pmap6.get(name).get(5)*Pmap6.get(name).get(6)*Pmap6.get(name).get(7)*Pmap6.get(name).get(8)*Pmap6.get(name).get(9)*Pmap6.get(name).get(10)/
						 (Pmap6.get(name).get(0)+Pmap6.get(name).get(1)+Pmap6.get(name).get(2)+Pmap6.get(name).get(3)+Pmap6.get(name).get(4)+Pmap6.get(name).get(5)+Pmap6.get(name).get(6)+Pmap6.get(name).get(7)+Pmap6.get(name).get(8)+Pmap6.get(name).get(9)+Pmap6.get(name).get(10))*(double)EightMap.size()/(double)list.size();
				 //store each result of probability
				 List<Double> p=new ArrayList<Double>();
				 p.add(p1);
				 p.add(p2);
				 p.add(p3);
				 p.add(p4);
				 p.add(p5);
				 p.add(p6);
				 //sort it to find the largest one
				 Collections.sort(p);
				 //the result will be the largest probability and store in the group based on the true quality
				 //for example, if the true quality is "Three" and the largest probability is p6 (quality "Eight"), the count in group "Eight" will add one which is actually in group "Three"
				 if (name.contains("Three")){if(p.get(5)==p1){three.set(0, three.get(0)+1);}
				 							 if(p.get(5)==p2){three.set(1, three.get(1)+1);}
				 							 if(p.get(5)==p3){three.set(2, three.get(2)+1);}
				 							 if(p.get(5)==p4){three.set(3, three.get(3)+1);}
				 							 if(p.get(5)==p5){three.set(4, three.get(4)+1);}
				 							 if(p.get(5)==p6){three.set(5, three.get(5)+1);}}			 
				
				 if (name.contains("Four")){if(p.get(5)==p1){four.set(0, four.get(0)+1);}
				 							if(p.get(5)==p2){four.set(1, four.get(1)+1);}
				 							if(p.get(5)==p4){four.set(3, four.get(3)+1);}
				 							if(p.get(5)==p5){four.set(4, four.get(4)+1);}
				 							if(p.get(5)==p6){four.set(5, four.get(5)+1);}}				 
				
				 if (name.contains("Five")){if(p.get(5)==p1){five.set(0, five.get(0)+1);}
				 							if(p.get(5)==p2){five.set(1, five.get(1)+1);}
				 							if(p.get(5)==p3){five.set(2, five.get(2)+1);}
				 							if(p.get(5)==p4){five.set(3, five.get(3)+1);}
				 							if(p.get(5)==p5){five.set(4, five.get(4)+1);}
				 							if(p.get(5)==p6){five.set(5, five.get(5)+1);}}
					
				 if (name.contains("Six")){if(p.get(5)==p1){six.set(0, six.get(0)+1);}
				 						   if(p.get(5)==p2){six.set(1, six.get(1)+1);}
				 						   if(p.get(5)==p3){six.set(2, six.get(2)+1);}
				 						   if(p.get(5)==p4){six.set(3, six.get(3)+1);}
				 						   if(p.get(5)==p5){six.set(4, six.get(4)+1);}
				 						   if(p.get(5)==p6){six.set(5, six.get(5)+1);}}
					
				 if (name.contains("Seven")){if(p.get(5)==p1){seven.set(0, seven.get(0)+1);}
				 							 if(p.get(5)==p2){seven.set(1, seven.get(1)+1);}
				 							 if(p.get(5)==p3){seven.set(2, seven.get(2)+1);}
				 							 if(p.get(5)==p4){seven.set(3, seven.get(3)+1);}
				 							 if(p.get(5)==p5){seven.set(4, seven.get(4)+1);}
				 							 if(p.get(5)==p6){seven.set(5, seven.get(5)+1);}}
					 
				 if (name.contains("Eight")){if(p.get(5)==p1){eight.set(0, eight.get(0)+1);}
				 							if(p.get(5)==p2){eight.set(1, eight.get(1)+1);}
				 							if(p.get(5)==p3){eight.set(2, eight.get(2)+1);}
				 							if(p.get(5)==p4){eight.set(3, eight.get(3)+1);}
				 							if(p.get(5)==p5){eight.set(4, eight.get(4)+1);}
				 							if(p.get(5)==p6){eight.set(5, eight.get(5)+1);}}
				 //if the result is same as the true quality, sum will add one
				 if (p.get(5)==p1&&name.contains("Three")||p.get(5)==p2&&name.contains("Four")||p.get(5)==p3&&name.contains("Five")||p.get(5)==p4&&name.contains("Six")||p.get(5)==p5&&name.contains("Seven")||p.get(5)==p6&&name.contains("Eight")){ 
					 sum=sum+1; 
				 }
			 }
			 //calculate accuracy by (number of right group samples/number of test samples) at each time
			 accuracy=sum/key.size();
			 wholesum+=accuracy;
			 //print confusion matrix at each time
			 System.out.println("confusion matrix-----------------------------------------");
			 System.out.println("      Three Four Five Six Seven Eight");
			 System.out.println("Three "+three);
			 System.out.println("Four  "+four);
			 System.out.println("Five  "+five);
			 System.out.println("Six   "+six);
			 System.out.println("Seven "+seven);
			 System.out.println("Eight "+eight);
			 System.out.println();
			 //print accuracy at each time
			 System.out.println("accuracy= "+accuracy);
			 System.out.println();
		 }
		 //calculate whole accuracy for 10 times
		 wholeaccuracy=wholesum/10;	
		 System.out.println("-------------------------------------------------------------");
		 //print whole accuracy
		 System.out.println("wholeaccuracy= "+wholeaccuracy);	 		
	}
		
}




