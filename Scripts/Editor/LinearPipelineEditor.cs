#if UNITY_EDITOR

using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections.Generic;

[CustomEditor(typeof(LinearPipeline))]
public class LinearPipelineEditor : Editor
{
  private const string MATERIAL_SUFFIX = "_Mat";
  private const string TEXTURE_SUFFIX = "_Tex";

  private SerializedProperty pipelineGeneratedPathProperty;
  private SerializedProperty initialStateProperty;

  private static Shader shader;
  private static int radix = 16;
  private static int fftResolution = 256;
  private static bool isInverse = false;
  private static bool doBothFFTAndInverse = false;
  private static int preFFTStages = 0;
  private static int postFFTStages = 0;

  private void OnEnable()
  {
    pipelineGeneratedPathProperty = serializedObject.FindProperty("pipelineGeneratedPath");

    initialStateProperty = serializedObject.FindProperty("initialState");

    // Set default shader if not already set
    if (shader == null)
    {
      shader = Shader.Find("yum_food/fft");
    }
  }

  public override void OnInspectorGUI()
  {
    serializedObject.Update();

    EditorGUILayout.PropertyField(serializedObject.FindProperty("materials"), true);
    EditorGUILayout.PropertyField(serializedObject.FindProperty("renderTextures"), true);

    LinearPipeline pipeline = (LinearPipeline)target;

    EditorGUILayout.Space();
    EditorGUILayout.LabelField("Pipeline Generation", EditorStyles.boldLabel);

    EditorGUILayout.PropertyField(pipelineGeneratedPathProperty);
    EditorGUILayout.PropertyField(initialStateProperty);

    serializedObject.ApplyModifiedProperties();

    shader = EditorGUILayout.ObjectField("Shader", shader, typeof(Shader), false) as Shader;
    radix = Mathf.Max(2, EditorGUILayout.IntField("Radix", radix));
    fftResolution = Mathf.Max(2, EditorGUILayout.IntField("FFT Resolution (N)", fftResolution));

    EditorGUI.BeginDisabledGroup(doBothFFTAndInverse);
    isInverse = EditorGUILayout.Toggle("Inverse FFT", isInverse);
    EditorGUI.EndDisabledGroup();

    doBothFFTAndInverse = EditorGUILayout.Toggle("Both FFT and Inverse FFT", doBothFFTAndInverse);
    preFFTStages = Mathf.Max(0, EditorGUILayout.IntField("Pre-FFT Stages", preFFTStages));
    postFFTStages = Mathf.Max(0, EditorGUILayout.IntField("Post-FFT Stages", postFFTStages));

    int fftStages = CalculateFFTStages(fftResolution, radix);
    EditorGUILayout.HelpBox($"FFT stages: {fftStages} (including bit reverse)", MessageType.Info);

    EditorGUILayout.Space();

    GUI.enabled = shader != null;
    if (GUILayout.Button("Create Pipeline", GUILayout.Height(30)))
    {
      CreatePipeline(pipeline);
    }
    GUI.enabled = true;

    if (shader == null)
    {
      EditorGUILayout.HelpBox("Please specify a shader to create the pipeline.", MessageType.Warning);
    }
  }

  private int CalculateFFTStages(int n, int radix)
  {
    return Mathf.CeilToInt(Mathf.Log(n) / Mathf.Log(radix)) * 2 + 1;
  }

  private void CreatePipeline(LinearPipeline pipeline)
  {
    string pipelineName = pipeline.gameObject.name;
    string pipelinePath = Path.Combine(pipeline.pipelineGeneratedPath, pipelineName);

    // Ensure directories exist
    if (!AssetDatabase.IsValidFolder(pipeline.pipelineGeneratedPath))
    {
      CreateFolderRecursive(pipeline.pipelineGeneratedPath);
    }

    if (!AssetDatabase.IsValidFolder(pipelinePath))
    {
      AssetDatabase.CreateFolder(pipeline.pipelineGeneratedPath, pipelineName);
    }

    Undo.RegisterFullObjectHierarchyUndo(pipeline.gameObject, "Create Pipeline");

    // Clear existing children
    ClearChildren(pipeline.transform);

    // Generate stage names
    List<string> stageNames = GenerateStageNames();

    // Create arrays for materials and textures
    Material[] materials = new Material[stageNames.Count];
    RenderTexture[] textures = new RenderTexture[stageNames.Count];

    // Create each stage
    for (int i = 0; i < stageNames.Count; i++)
    {
      CreateStage(pipeline.transform, stageNames[i], pipelinePath, i, ref materials, ref textures);
    }

    // Assign textures to materials
    AssignTexturesToMaterials(materials, textures, pipeline.initialState);

    // Update pipeline component
    pipeline.materials = materials;
    pipeline.renderTextures = textures;

    EditorUtility.SetDirty(pipeline);
    AssetDatabase.SaveAssets();

    Debug.Log($"[LinearPipeline] Created pipeline '{pipelineName}' with {materials.Length} stages");
  }

  private List<string> GenerateStageNames()
  {
    List<string> names = new List<string>();
    int totalFFTStages = CalculateFFTStages(fftResolution, radix);

    // Pre-FFT stages
    for (int i = 0; i < preFFTStages; i++)
    {
      names.Add($"Pre_Stage_{i:D2}");
    }

    // FFT stages
    if (doBothFFTAndInverse || !isInverse)
    {
      for (int i = 0; i < totalFFTStages; i++)
      {
        names.Add($"FFT_Stage_{i:D2}");
      }
    }

    // Inverse FFT stages
    if (doBothFFTAndInverse || isInverse)
    {
      for (int i = 0; i < totalFFTStages; i++)
      {
        names.Add($"IFFT_Stage_{i:D2}");
      }
    }

    // Post-FFT stages
    for (int i = 0; i < postFFTStages; i++)
    {
      names.Add($"Post_Stage_{i:D2}");
    }

    return names;
  }

  private void CreateStage(Transform parent, string stageName, string basePath, int index,
    ref Material[] materials, ref RenderTexture[] textures)
  {
    // Create stage GameObject as a quad
    GameObject stageGO = GameObject.CreatePrimitive(PrimitiveType.Quad);
    stageGO.name = stageName;
    stageGO.transform.SetParent(parent);
    stageGO.transform.localPosition = new Vector3(index * 1.0f, 0, 0);  // Space 1 meter apart on X axis
    stageGO.transform.localRotation = Quaternion.identity;
    stageGO.transform.localScale = Vector3.one;

    DestroyImmediate(stageGO.GetComponent<MeshCollider>());

    // Create or update material
    string materialPath = $"{basePath}/{stageName}{MATERIAL_SUFFIX}.mat";
    Material material = AssetDatabase.LoadAssetAtPath<Material>(materialPath);

    if (material == null)
    {
      material = new Material(shader);
      AssetDatabase.CreateAsset(material, materialPath);
    }
    else
    {
      material.shader = shader;
      EditorUtility.SetDirty(material);
    }

    // Set shader properties based on stage type
    ConfigureMaterialProperties(material, stageName);

    // Create or update render texture
    string texturePath = $"{basePath}/{stageName}{TEXTURE_SUFFIX}.renderTexture";
    RenderTexture texture = AssetDatabase.LoadAssetAtPath<RenderTexture>(texturePath);

    if (texture == null)
    {
      texture = new RenderTexture(fftResolution, fftResolution, 0, RenderTextureFormat.ARGBFloat)
      {
        filterMode = FilterMode.Point,
        wrapMode = TextureWrapMode.Clamp
      };
      AssetDatabase.CreateAsset(texture, texturePath);
    }
    else
    {
      texture.width = fftResolution;
      texture.height = fftResolution;
      EditorUtility.SetDirty(texture);
    }

    // Assign material to renderer
    MeshRenderer renderer = stageGO.GetComponent<MeshRenderer>();
    renderer.sharedMaterial = material;

    // Store in arrays
    materials[index] = material;
    textures[index] = texture;
  }

  private void ConfigureMaterialProperties(Material material, string stageName)
  {
    // Set common properties
    material.SetInt("_N", fftResolution);
    material.SetInt("_Radix", radix);

    // Reset all flags
    material.SetFloat("_Passthrough", 0f);
    material.SetFloat("_LDS", 0f);
    material.SetFloat("_Luminance", 0f);
    material.SetFloat("_Inverse", 0f);
    material.SetFloat("_BitReversal", 0f);

    // Configure based on stage type
    if (stageName.StartsWith("Pre_Stage_"))
    {
      // Pre-processing stages - set as passthrough
      material.SetFloat("_Passthrough", 1f);
      material.SetInt("_Stage", 0);
    }
    else if (stageName.StartsWith("FFT_Stage_"))
    {
      // Extract stage number
      string stageNumStr = stageName.Replace("FFT_Stage_", "");
      if (int.TryParse(stageNumStr, out int stageNum))
      {
        material.SetInt("_Stage", stageNum);

        // Last stage is bit reversal
        int totalStages = CalculateFFTStages(fftResolution, radix);
        if (stageNum == totalStages - 1)
        {
          material.SetFloat("_BitReversal", 1f);
        }
      }
      material.SetFloat("_Inverse", 0f);
    }
    else if (stageName.StartsWith("IFFT_Stage_"))
    {
      // Extract stage number
      string stageNumStr = stageName.Replace("IFFT_Stage_", "");
      if (int.TryParse(stageNumStr, out int stageNum))
      {
        material.SetInt("_Stage", stageNum);

        // Last stage is bit reversal
        int totalStages = CalculateFFTStages(fftResolution, radix);
        if (stageNum == totalStages - 1)
        {
          material.SetFloat("_BitReversal", 1f);
        }
      }
      material.SetFloat("_Inverse", 1f);
    }
    else if (stageName.StartsWith("Post_Stage_"))
    {
      // Post-processing stages - set as passthrough
      material.SetFloat("_Passthrough", 1f);
      material.SetInt("_Stage", 0);
    }

    EditorUtility.SetDirty(material);
  }

  private void AssignTexturesToMaterials(Material[] materials, RenderTexture[] textures, Texture initialState)
  {
    for (int i = 0; i < materials.Length; i++)
    {
      Texture inputTexture = (i == 0) ? initialState : textures[i - 1];

      if (inputTexture != null)
      {
        materials[i].SetTexture("_MainTex", inputTexture);
      }

      EditorUtility.SetDirty(materials[i]);
    }
  }

  private void ClearChildren(Transform parent)
  {
    while (parent.childCount > 0)
    {
      DestroyImmediate(parent.GetChild(0).gameObject);
    }
  }

  private void CreateFolderRecursive(string path)
  {
    string[] folders = path.Split('/');
    string currentPath = folders[0];

    for (int i = 1; i < folders.Length; i++)
    {
      string nextPath = Path.Combine(currentPath, folders[i]);
      if (!AssetDatabase.IsValidFolder(nextPath))
      {
        AssetDatabase.CreateFolder(currentPath, folders[i]);
      }
      currentPath = nextPath;
    }
  }
}

#endif
