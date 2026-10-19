package ai.greatfree.distributed.cnn.message;

import java.util.List;

import edu.greatfree.cluster.message.ClusterRequest;


public class DistributeTrainingDataRequest extends ClusterRequest
{
	private static final long serialVersionUID = 6632905411177612028L;

	private String childKey;
	private String fileName;
	private List<String> classNames;
	private List<byte[]> images;
	private List<Integer> labels;

	public DistributeTrainingDataRequest(String childKey, String fileName, List<String> classNames, List<byte[]> images, List<Integer> labels)
	{
		super(CNNAppID.DISTRIBUTE_TRAINING_DATA_REQUEST, childKey);
		this.childKey = childKey;
		this.fileName = fileName;
		this.classNames =classNames;
		this.images = images;
		this.labels = labels;
		
	}
	
	public String getChildKey()
	{
		return this.childKey;
	}
	
	public String getFileName()
	{
		return this.fileName;
	}
	
	public List<byte[]> getImages()
	{
		return this.images;
	}

	public List<Integer> getLabels() {
		return labels;
	}

	public void setLabels(List<Integer> labels) {
		this.labels = labels;
	}

	public List<String> getClassNames() {
		return classNames;
	}

	public void setClassNames(List<String> classNames) {
		this.classNames = classNames;
	}
}
