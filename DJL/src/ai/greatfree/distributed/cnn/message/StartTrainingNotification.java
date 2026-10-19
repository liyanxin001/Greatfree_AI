package ai.greatfree.distributed.cnn.message;

import edu.greatfree.cluster.message.ClusterNotification;
import edu.greatfree.cluster.message.NotificationType;

/**
 * Created: 08/29/2024, Bing Li
 */
public class StartTrainingNotification extends ClusterNotification
{
	private static final long serialVersionUID = -5146341560525948060L;

	/*
	public StartTrainingNotification(String childKey)
	{
		super(RNNAppID.START_TRAINING_NOTIFICATION, childKey);
	}
	*/

	public StartTrainingNotification()
	{
		super(NotificationType.BROADCAST_NOTIFICATION, CNNAppID.START_TRAINING_NOTIFICATION);
	}
}

