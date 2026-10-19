package ai.greatfree.distributed.cnn.message;

import edu.greatfree.container.message.Notification;

/**
 * Created: 09/01/2024, Bing Li
 */
public class TrainingDoneNotification extends Notification
{
	private static final long serialVersionUID = -4106024072222268503L;
	
	private String childKey;

	public TrainingDoneNotification(String childKey)
	{
		super(CNNAppID.TRAINING_DONE_NOTIFICATION);
		this.childKey = childKey;
	}

	public String getChildKey()
	{
		return this.childKey;
	}
}

