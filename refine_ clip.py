def refine_prompt(prompt):
    """
    Refines the prompt by adding prefixes, suffixes, and style modifiers to improve its suitability for CLIP.
    """
    # Clean up the prompt
    prompt = prompt.strip()

    # Define possible prefixes
    prefixes = [
        "A high-quality photo of",
        "An ultra-detailed painting of",
        "A beautiful illustration of",
        "A cinematic shot of",
        "A hyper-realistic image of",
        "A professional photograph of",
        "A stunning digital artwork of",
        "An 8K resolution render of",
    ]

    # Define possible suffixes
    suffixes = [
        "trending on ArtStation",
        "concept art",
        "unreal engine",
        "octane render",
        "award-winning",
        "ultra-realistic",
        "hyper-detailed",
        "photo-realistic",
    ]

    # Define possible styles
    styles = [
        "in the style of Vincent van Gogh",
        "by Leonardo da Vinci",
        "in the style of Studio Ghibli",
        "by Greg Rutkowski",
        "digital art",
        "fantasy art",
        "cyberpunk style",
        "futuristic",
    ]

    # You can customize how prefixes, suffixes, and styles are selected
    prefix = prefixes[0]  # For example, select the first prefix
    suffix = suffixes[0]  # Select the first suffix
    style = styles[0]     # Select the first style

    # Combine them to form the refined prompt
    refined_prompt = f"{prefix} {prompt}, {suffix}, {style}"

    return refined_prompt