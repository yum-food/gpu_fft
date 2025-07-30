using UdonSharp;
using UnityEngine;
using VRC.SDKBase;

[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public class LinearPipeline : UdonSharpBehaviour
{
  [Header("Pipeline Assets")]
  public string pipelineGeneratedPath = "Assets/yum_food/gpu_fft/Pipeline_Generated";
  public Texture initialState;
  public Material[] materials;
  public RenderTexture[] renderTextures;

  private bool isValid;

  void Start()
  {
    ValidatePipeline();
  }

  private void ValidatePipeline()
  {
    isValid = materials != null &&
              renderTextures != null &&
              materials.Length > 0 &&
              materials.Length == renderTextures.Length;

    if (!isValid)
    {
      Debug.LogError($"[LinearPipeline] Invalid configuration on {gameObject.name}");
    }
  }

  public void RunPipeline()
  {
    if (!isValid || initialState == null) return;

    VRCGraphics.Blit(initialState, renderTextures[0], materials[0], -1);

    for (int i = 1; i < materials.Length; i++)
    {
      VRCGraphics.Blit(renderTextures[i-1], renderTextures[i], materials[i], -1);
    }
  }
}
