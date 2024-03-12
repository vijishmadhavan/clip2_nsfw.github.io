class NsfwDetector {
    constructor() {
        this._threshold = 0.2;
        this._nsfwLabels = [
            'FEMALE_BREAST_EXPOSED',
            'FEMALE_GENITALIA_EXPOSED',
            'BUTTOCKS_EXPOSED',
            'ANUS_EXPOSED',
            'MALE_GENITALIA_EXPOSED',
            'BLOOD_SHED',
            'VIOLENCE',
            'GORE',
            'PORNOGRAPHY',
            'DRUGS',
            'ALCOHOL',
        ];
    }

    async isNsfw(imageUrl) {
        try {
            const classifier = await window.tensorflowPipeline('zero-shot-image-classification', 'Xenova/clip-vit-base-patch32');
            const output = await classifier(imageUrl, this._nsfwLabels);
            console.log(output);
            const nsfwDetected = output.some(result => result.score > this._threshold);
            return nsfwDetected;
        } catch (error) {
            console.error('Error during NSFW classification: ', error);
            throw error;
        }
    }
}

// Attach the NsfwDetector class to the window object to make it globally accessible
window.NsfwDetector = NsfwDetector;

