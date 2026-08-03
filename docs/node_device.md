# Low-Cost Urban Acoustic Monitoring Device

## Original Hardware (2016) vs. Modern Equivalent (2026)

### Reference

Mydlarz, C., Salamon, J., & Bello, J. P. (2016). *The Implementation of Low-cost Urban Acoustic Monitoring Devices* (arXiv:1605.08450).

---

# 1. Original Implementation (2016)

The device presented in the paper was designed as a low-cost autonomous environmental acoustic monitoring station capable of continuously recording and transmitting urban sound data.

## Main Components

| Component                | Original Part                          | Purpose                                                                                                     |
| ------------------------ | -------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Processing Unit          | Raspberry Pi Model B+                  | Main computer running Linux, recording audio, data management, and networking                               |
| Microphone               | Knowles SPU0410LR5H-QB MEMS microphone | Digital MEMS microphone for sound acquisition                                                               |
| Microphone Interface PCB | Custom-designed PCB                    | Supplies power, clocks the microphone, converts PDM output to PCM/I²S, and interfaces with the Raspberry Pi |
| Storage                  | microSD Card                           | Operating system and local storage                                                                          |
| Networking               | USB Wi-Fi Adapter                      | Wireless communication with the server                                                                      |
| Power Supply             | 5 V regulated supply                   | Powers the Raspberry Pi and microphone electronics                                                          |
| Enclosure                | Weatherproof enclosure                 | Outdoor protection                                                                                          |
| Acoustic Port            | Protected microphone opening           | Allows sound to reach the microphone while minimizing environmental damage                                  |
| Mounting Hardware        | Pole/building mount                    | Outdoor installation                                                                                        |

---

## Microphone Electronics

The custom PCB included:

* Knowles SPU0410LR5H-QB MEMS microphone
* Clean power supply
* Clock generation
* PDM interface
* PDM-to-PCM conversion
* I²S audio interface to the Raspberry Pi

Unlike conventional measurement microphones, the authors designed their own electronics to maintain full control over the audio signal path and enable accurate calibration.

---

## Calibration Equipment

The monitoring nodes were calibrated against professional acoustic equipment, including:

* Class 1 Sound Level Meter
* Acoustic calibrator (typically 94 dB @ 1 kHz)

---

# 2. Modern Equivalent (2026)

Most of the original components have now been replaced by newer hardware that offers improved performance while reducing complexity and cost.

## Recommended Components

| Component            | Recommended Part                                                                              | Notes                                                         |
| -------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| Processing Unit      | Raspberry Pi 5 (or Raspberry Pi Zero 2 W for lower power)                                     | Built-in Wi-Fi and significantly faster processing            |
| Microphone           | TDK InvenSense ICS-40720 (analog), ICS-40619 (analog), or Knowles SPH0645LM4H-B (digital I²S) | Stable MEMS microphones suitable for acoustic sensing         |
| Microphone Interface | Off-the-shelf I²S MEMS breakout board (Adafruit, SparkFun, etc.)                              | Eliminates the need for a custom PCB in many applications     |
| Storage              | High-Endurance microSD Card (32–128 GB)                                                       | Better reliability for continuous recording                   |
| Networking           | Integrated Wi-Fi                                                                              | No external USB adapter required                              |
| Power Supply         | 5 V USB-C regulated supply                                                                    | Standard Raspberry Pi power input                             |
| Enclosure            | IP65/IP66 weatherproof enclosure                                                              | Outdoor protection                                            |
| Acoustic Vent        | Waterproof acoustic membrane (e.g., Gore vent)                                                | Protects the microphone while preserving acoustic performance |
| Mounting             | Stainless-steel or aluminum mounting bracket                                                  | Outdoor deployment                                            |

---

## Optional Environmental Sensors

To extend the monitoring capabilities, the following sensors can be added:

* Temperature sensor
* Relative humidity sensor
* Atmospheric pressure sensor
* GPS receiver (for synchronization and location)
* Air quality sensors (PM2.5, PM10, CO₂, VOC)
* Wind speed and direction sensors

---

# 3. Software Stack

The modern implementation would typically include:

* Raspberry Pi OS Lite
* Python-based acquisition software
* ALSA or PipeWire audio interface
* Remote SSH administration
* Automatic data upload (SFTP, MQTT, HTTPS, or cloud storage)
* Local buffering in case of network outages
* System health monitoring
* Automatic software updates

---

# 4. Suggested Bill of Materials (Modern)

* Raspberry Pi 5 (4 GB) *(or Raspberry Pi Zero 2 W for a lower-cost deployment)*
* High-Endurance microSD card
* MEMS microphone breakout board (I²S or analog, depending on design)
* Weatherproof IP65/IP66 enclosure
* Waterproof acoustic vent
* USB-C power supply or PoE HAT (optional)
* Mounting bracket and fasteners
* Weather-resistant cable glands
* Optional battery backup or solar power system for remote installations

---

# 5. Key Improvements Over the Original Design

* No external Wi-Fi adapter required.
* Higher processing performance enables edge AI and real-time sound classification.
* Simpler hardware through commercially available microphone breakout boards.
* Lower power consumption (especially when using Raspberry Pi Zero 2 W).
* Improved storage reliability with high-endurance microSD cards.
* Easier maintenance due to the widespread availability of modern components.
* Optional integration of environmental sensors and cloud connectivity without significant hardware redesign.

---

# 6. Notes

If the goal is **scientific or regulatory-grade environmental noise measurements**, the microphone subsystem should still be calibrated against a certified sound level meter and acoustic calibrator, as described in the original paper. For research applications requiring IEC 61672 compliance, additional attention should be given to microphone selection, enclosure design, frequency response characterization, and calibration procedures.
