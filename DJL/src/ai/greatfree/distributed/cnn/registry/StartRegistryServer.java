package ai.greatfree.distributed.cnn.registry;

/*    */ import org.greatfree.util.TerminateSignal;
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ class StartRegistryServer
/*    */ {
/*    */   public static void main(String[] args) {
/* 15 */     System.out.println("Registry server starting up ...");
/*    */ 
/*    */     
/*    */     try {
/* 19 */       RegisterServer.CS().start(8941, new RegistryTask());
/*    */     }
/* 21 */     catch (ClassNotFoundException|java.io.IOException|org.greatfree.exceptions.RemoteReadException e) {
/*    */       
/* 23 */       e.printStackTrace();
/*    */     } 
/*    */     
/* 26 */     System.out.println("Registry server started ...");
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */     
/* 43 */     TerminateSignal.SIGNAL().waitTermination();
/*    */   }
/*    */ }
