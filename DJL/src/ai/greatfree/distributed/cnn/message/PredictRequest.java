package ai.greatfree.distributed.cnn.message;

import edu.greatfree.cluster.message.ClusterRequest;
import edu.greatfree.cluster.message.RequestType;

/**
 * Created: 08/29/2024, Bing Li
 */
public class PredictRequest extends ClusterRequest
{
	private static final long serialVersionUID = 7866978416006800710L;
	
	private byte[] imageBytes;
	private String fileName;


	public PredictRequest(byte[] imageBytes, String fileName)
	{
		super(RequestType.BROADCAST_REQUEST, CNNAppID.PREDICT_REQUEST);
		this.imageBytes = imageBytes;
		this.fileName = fileName;
	}

	public byte[] getImageBytes()
	{
		return this.imageBytes;
	}
	
	public String getFileName() 
	{
		return this.fileName;
	}
	


}

